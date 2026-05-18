"""Tests for the LLM ticket summariser. LLM client is mocked — no API calls."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from ticketrouting.models.summary.summarizer import (
    SUMMARY_RUBRIC_VERSION,
    TicketSummarizer,
)
from ticketrouting.schemas import Entities, Sentiment


@dataclass
class FakeClient:
    """Implements the LLMClient protocol; counts calls so we can assert on caching."""

    response: str = json.dumps(
        {
            "summary": "Enterprise customer reports duplicate charges on the Scale plan.",
            "entities": {
                "product": "Scale plan",
                "error_codes": [],
                "sentiment": "frustrated",
                "deadline_mentioned": "before end of billing cycle",
            },
        }
    )
    model: str = "fake-model-v1"
    calls: int = 0
    seen_users: list[str] = field(default_factory=list)

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        self.seen_users.append(user)
        return self.response


def test_summarize_returns_structured_result(tmp_path):
    client = FakeClient()
    summarizer = TicketSummarizer(client=client, cache_dir=tmp_path)

    result = summarizer.summarize("We were charged twice on our Scale plan.")

    assert "duplicate charges" in result.summary
    assert result.entities.product == "Scale plan"
    assert result.entities.sentiment == Sentiment.FRUSTRATED
    assert result.entities.error_codes == []
    assert result.entities.deadline_mentioned == "before end of billing cycle"
    assert result.rubric_version == SUMMARY_RUBRIC_VERSION
    assert result.model == "fake-model-v1"
    assert client.calls == 1


def test_cache_skips_second_call(tmp_path):
    client = FakeClient()
    summarizer = TicketSummarizer(client=client, cache_dir=tmp_path)

    summarizer.summarize("identical ticket")
    summarizer.summarize("identical ticket")

    assert client.calls == 1


def test_cache_round_trips_entities_fields(tmp_path):
    """Reading from cache must reconstruct the same typed entities, not a dict."""
    client = FakeClient(
        response=json.dumps(
            {
                "summary": "App crashed after the v2 update.",
                "entities": {
                    "product": "MobileApp",
                    "error_codes": ["E_NULL_REF", "500"],
                    "sentiment": "angry",
                    "deadline_mentioned": None,
                },
            }
        )
    )
    summarizer = TicketSummarizer(client=client, cache_dir=tmp_path)

    summarizer.summarize("ticket-text")

    # Fresh instance reads off disk only — no LLM call should happen.
    fresh_client = FakeClient()
    fresh = TicketSummarizer(client=fresh_client, cache_dir=tmp_path)
    result = fresh.summarize("ticket-text")

    assert fresh_client.calls == 0
    assert result.entities.product == "MobileApp"
    assert result.entities.error_codes == ["E_NULL_REF", "500"]
    assert result.entities.sentiment == Sentiment.ANGRY
    assert result.entities.deadline_mentioned is None


def test_handles_missing_optional_fields(tmp_path):
    """Null product/deadline + empty error_codes is a legitimate response."""
    client = FakeClient(
        response=json.dumps(
            {
                "summary": "Customer asks how to update billing email.",
                "entities": {
                    "product": None,
                    "error_codes": [],
                    "sentiment": "neutral",
                    "deadline_mentioned": None,
                },
            }
        )
    )
    summarizer = TicketSummarizer(client=client, cache_dir=tmp_path)

    result = summarizer.summarize("how do I change my billing email?")

    assert result.entities.product is None
    assert result.entities.deadline_mentioned is None
    assert result.entities.sentiment == Sentiment.NEUTRAL


def test_normalises_sentiment_case(tmp_path):
    """LLM occasionally returns 'Frustrated' instead of 'frustrated'."""
    client = FakeClient(
        response=json.dumps(
            {
                "summary": "x",
                "entities": {
                    "product": None,
                    "error_codes": [],
                    "sentiment": "Frustrated",
                    "deadline_mentioned": None,
                },
            }
        )
    )
    summarizer = TicketSummarizer(client=client, cache_dir=tmp_path)

    result = summarizer.summarize("x")

    assert result.entities.sentiment == Sentiment.FRUSTRATED


def test_coerces_null_error_codes_to_empty_list(tmp_path):
    """Sometimes the LLM emits error_codes: null instead of []."""
    client = FakeClient(
        response=json.dumps(
            {
                "summary": "x",
                "entities": {
                    "product": None,
                    "error_codes": None,
                    "sentiment": "neutral",
                    "deadline_mentioned": None,
                },
            }
        )
    )
    summarizer = TicketSummarizer(client=client, cache_dir=tmp_path)

    result = summarizer.summarize("x")

    assert result.entities.error_codes == []


def test_strips_markdown_code_fences(tmp_path):
    payload = {
        "summary": "y",
        "entities": {
            "product": None,
            "error_codes": [],
            "sentiment": "positive",
            "deadline_mentioned": None,
        },
    }
    client = FakeClient(response=f"```json\n{json.dumps(payload)}\n```")
    summarizer = TicketSummarizer(client=client, cache_dir=tmp_path)

    result = summarizer.summarize("y")

    assert result.summary == "y"
    assert result.entities.sentiment == Sentiment.POSITIVE


def test_invalid_sentiment_raises(tmp_path):
    """Sentiment outside the enum must blow up — silent coercion would mask LLM drift."""
    client = FakeClient(
        response=json.dumps(
            {
                "summary": "x",
                "entities": {
                    "product": None,
                    "error_codes": [],
                    "sentiment": "ecstatic",
                    "deadline_mentioned": None,
                },
            }
        )
    )
    summarizer = TicketSummarizer(client=client, cache_dir=tmp_path)

    with pytest.raises(Exception):  # pydantic ValidationError
        summarizer.summarize("x")


def test_does_not_collide_with_label_caches(tmp_path):
    """Summary, urgency, and category labelers can share the cache dir.

    Each one's rubric_version namespaces the sha256 key, so this is just a
    sanity-check that no surprise file-name collision pops up if all three are
    pointed at the same directory in production.
    """
    from ticketrouting.data.label import CategoryLabeler, UrgencyLabeler

    text = "I can't log in"

    @dataclass
    class _Client:
        response: str
        model: str = "shared-model"
        calls: int = 0

        def complete(self, system: str, user: str) -> str:
            self.calls += 1
            return self.response

    urgency = _Client(response='{"urgency": "high", "reasoning": "locked"}')
    category = _Client(response='{"category": "account", "reasoning": "login"}')
    summarizer = _Client(
        response=json.dumps(
            {
                "summary": "Login failure.",
                "entities": {
                    "product": None,
                    "error_codes": [],
                    "sentiment": "frustrated",
                    "deadline_mentioned": None,
                },
            }
        )
    )

    UrgencyLabeler(client=urgency, cache_dir=tmp_path).label(text)
    CategoryLabeler(client=category, cache_dir=tmp_path).label(text)
    result = TicketSummarizer(client=summarizer, cache_dir=tmp_path).summarize(text)

    # Each must have hit its own client exactly once (no cache cross-talk).
    assert urgency.calls == 1
    assert category.calls == 1
    assert summarizer.calls == 1
    assert result.summary == "Login failure."
    assert isinstance(result.entities, Entities)
