"""Tests for the RoutingService + the FastAPI app.

Strategy:
- The service tests run RoutingService directly with fake model objects so we
  exercise the orchestration logic (parallelism, deadline, graceful
  degradation) without any sklearn or LLM dependencies.
- The API tests use FastAPI's TestClient with the get_service dependency
  overridden, so we verify the HTTP contract and DI plumbing without
  triggering real model loads.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from ticketrouting.api.app import app, get_service
from ticketrouting.api.service import RoutingService
from ticketrouting.models.summary.summarizer import TicketSummary
from ticketrouting.schemas import Category, Entities, Sentiment, TicketIn, Urgency


# ----------------------------------------------------------------------------
# Fake model objects
# ----------------------------------------------------------------------------


@dataclass
class _FakeCategoryPipe:
    """Stand-in for a fitted sklearn pipeline. predict_one reads .classes_ + predict_proba."""

    category: str = "billing"
    confidence: float = 0.91

    @property
    def classes_(self):
        # predict_one indexes pipe.classes_ by argmax; we put our class at index 0.
        return [self.category]

    def predict_proba(self, X):
        import numpy as np

        return np.array([[self.confidence]] * len(X))


@dataclass
class _FakeUrgencyPipe:
    urgency: str = "high"
    confidence: float = 0.78

    @property
    def named_steps(self):
        return {"clf": _FakeUrgencyClf(self.urgency)}

    def predict_proba(self, X):
        import numpy as np

        return np.array([[self.confidence]] * len(X))


@dataclass
class _FakeUrgencyClf:
    urgency: str

    @property
    def classes_(self):
        return [self.urgency]


@dataclass
class _FakeSummarizer:
    result: TicketSummary | None = None
    delay_s: float = 0.0
    raise_exc: Exception | None = None
    calls: int = 0

    def summarize(self, text: str) -> TicketSummary:
        self.calls += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.raise_exc:
            raise self.raise_exc
        return self.result or _default_summary()


def _default_summary() -> TicketSummary:
    return TicketSummary(
        summary="Duplicate charges on Scale plan.",
        entities=Entities(
            product="Scale plan",
            error_codes=[],
            sentiment=Sentiment.FRUSTRATED,
            deadline_mentioned=None,
        ),
        model="fake",
        rubric_version="sum-v1",
    )


# ----------------------------------------------------------------------------
# RoutingService unit tests
# ----------------------------------------------------------------------------


def test_happy_path_assembles_full_decision():
    svc = RoutingService(
        category_pipe=_FakeCategoryPipe(category="billing", confidence=0.91),
        urgency_pipe=_FakeUrgencyPipe(urgency="high", confidence=0.78),
        summarizer=_FakeSummarizer(),
    )

    decision = asyncio.run(svc.route(TicketIn(text="we got charged twice")))

    assert decision.route_to == [Category.BILLING]
    assert decision.route_confidence == pytest.approx(0.91)
    assert decision.urgency == Urgency.HIGH
    assert decision.urgency_score == pytest.approx(0.78)
    assert decision.summary == "Duplicate charges on Scale plan."
    assert decision.entities is not None
    assert decision.entities.product == "Scale plan"
    assert decision.latency_ms is not None and decision.latency_ms >= 0


def test_missing_category_model_falls_back_to_other_with_zero_confidence():
    svc = RoutingService(category_pipe=None, urgency_pipe=_FakeUrgencyPipe())

    decision = asyncio.run(svc.route(TicketIn(text="x")))

    assert decision.route_to == [Category.OTHER]
    assert decision.route_confidence == 0.0


def test_missing_urgency_model_falls_back_to_low():
    svc = RoutingService(category_pipe=_FakeCategoryPipe(), urgency_pipe=None)

    decision = asyncio.run(svc.route(TicketIn(text="x")))

    assert decision.urgency == Urgency.LOW
    assert decision.urgency_score == 0.0


def test_missing_summarizer_leaves_summary_null():
    svc = RoutingService(
        category_pipe=_FakeCategoryPipe(),
        urgency_pipe=_FakeUrgencyPipe(),
        summarizer=None,
    )

    decision = asyncio.run(svc.route(TicketIn(text="x")))

    assert decision.summary is None
    assert decision.entities is None


def test_summarizer_exception_does_not_break_response():
    """Rate-limit or any LLM failure must degrade to summary=None, not 500."""
    svc = RoutingService(
        category_pipe=_FakeCategoryPipe(),
        urgency_pipe=_FakeUrgencyPipe(),
        summarizer=_FakeSummarizer(raise_exc=RuntimeError("groq 429")),
    )

    decision = asyncio.run(svc.route(TicketIn(text="x")))

    assert decision.route_to == [Category.BILLING]
    assert decision.summary is None
    assert decision.entities is None


def test_summary_deadline_drops_slow_summarizer():
    """If the summariser overruns the budget, return classification anyway."""
    svc = RoutingService(
        category_pipe=_FakeCategoryPipe(),
        urgency_pipe=_FakeUrgencyPipe(),
        summarizer=_FakeSummarizer(delay_s=0.5),
        summary_deadline_s=0.05,
    )

    decision = asyncio.run(svc.route(TicketIn(text="x")))

    assert decision.route_to == [Category.BILLING]
    assert decision.summary is None


def test_loaded_reports_each_component():
    svc = RoutingService(category_pipe=_FakeCategoryPipe(), urgency_pipe=None, summarizer=None)
    assert svc.loaded == {"category": True, "urgency": False, "summarizer": False}


def test_from_env_does_not_raise_when_artifacts_missing(tmp_path, monkeypatch):
    """from_env must produce a working service even if nothing is loadable."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    svc = RoutingService.from_env(
        category_path=tmp_path / "nope.joblib",
        urgency_path=tmp_path / "nope.joblib",
    )

    assert svc.loaded == {"category": False, "urgency": False, "summarizer": False}
    # And it can still respond.
    decision = asyncio.run(svc.route(TicketIn(text="hello")))
    assert decision.route_to == [Category.OTHER]
    assert decision.urgency == Urgency.LOW


# ----------------------------------------------------------------------------
# FastAPI HTTP tests
# ----------------------------------------------------------------------------


@pytest.fixture
def client_with_fake_service():
    """TestClient with a fully-mocked service injected via FastAPI DI."""
    svc = RoutingService(
        category_pipe=_FakeCategoryPipe(category="account", confidence=0.66),
        urgency_pipe=_FakeUrgencyPipe(urgency="medium", confidence=0.55),
        summarizer=_FakeSummarizer(),
    )
    app.dependency_overrides[get_service] = lambda: svc
    client = TestClient(app)
    try:
        yield client
    finally:
        app.dependency_overrides.clear()


def test_health_endpoint_reports_loaded_models(client_with_fake_service):
    resp = client_with_fake_service.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] == {"category": True, "urgency": True, "summarizer": True}
    assert body["fully_loaded"] is True


def test_route_endpoint_returns_routing_decision(client_with_fake_service):
    resp = client_with_fake_service.post(
        "/route", json={"text": "I can't log into my account"}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["route_to"] == ["account"]
    assert body["route_confidence"] == pytest.approx(0.66)
    assert body["urgency"] == "medium"
    assert body["summary"]  # non-empty
    assert body["entities"]["product"] == "Scale plan"
    assert isinstance(body["latency_ms"], int)


def test_route_rejects_empty_text(client_with_fake_service):
    """TicketIn requires text min_length=1 — pydantic should 422 the request."""
    resp = client_with_fake_service.post("/route", json={"text": ""})
    assert resp.status_code == 422


def test_route_accepts_optional_metadata(client_with_fake_service):
    resp = client_with_fake_service.post(
        "/route",
        json={
            "text": "Account locked",
            "ticket_id": "t-123",
            "customer_tier": "enterprise",
            "locale": "en-US",
        },
    )
    assert resp.status_code == 200
