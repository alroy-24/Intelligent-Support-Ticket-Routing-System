"""Tests for the urgency and category labelers. LLM client is mocked — no API calls."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from ticketrouting.data.label import (
    CATEGORY_RUBRIC_VERSION,
    RUBRIC_VERSION,
    CategoryLabeler,
    UrgencyLabeler,
)
from ticketrouting.schemas import Category, Urgency


@dataclass
class FakeClient:
    """Implements the LLMClient protocol; counts calls so we can assert on caching."""

    response: str = '{"urgency": "high", "reasoning": "account locked"}'
    model: str = "fake-model-v1"
    calls: int = 0
    seen_users: list[str] = field(default_factory=list)

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        self.seen_users.append(user)
        return self.response


def test_labels_ticket_with_valid_json(tmp_path):
    client = FakeClient()
    labeler = UrgencyLabeler(client=client, cache_dir=tmp_path)

    label = labeler.label("I can't log in to my account.")

    assert label.urgency == Urgency.HIGH
    assert label.reasoning == "account locked"
    assert label.rubric_version == RUBRIC_VERSION
    assert label.model == "fake-model-v1"
    assert client.calls == 1


def test_cache_skips_second_call_for_same_text(tmp_path):
    client = FakeClient()
    labeler = UrgencyLabeler(client=client, cache_dir=tmp_path)

    labeler.label("duplicate ticket")
    labeler.label("duplicate ticket")

    assert client.calls == 1, "second call should be served from cache"


def test_different_text_triggers_new_call(tmp_path):
    client = FakeClient()
    labeler = UrgencyLabeler(client=client, cache_dir=tmp_path)

    labeler.label("ticket A")
    labeler.label("ticket B")

    assert client.calls == 2


def test_strips_markdown_code_fences(tmp_path):
    client = FakeClient(
        response='```json\n{"urgency": "critical", "reasoning": "outage"}\n```'
    )
    labeler = UrgencyLabeler(client=client, cache_dir=tmp_path)

    label = labeler.label("the whole site is down")

    assert label.urgency == Urgency.CRITICAL


def test_invalid_json_raises(tmp_path):
    client = FakeClient(response="not json at all")
    labeler = UrgencyLabeler(client=client, cache_dir=tmp_path)

    with pytest.raises(json.JSONDecodeError):
        labeler.label("some ticket")


def test_cache_persists_across_labeler_instances(tmp_path):
    client_a = FakeClient()
    UrgencyLabeler(client=client_a, cache_dir=tmp_path).label("same text")

    client_b = FakeClient()  # fresh client, fresh call counter
    UrgencyLabeler(client=client_b, cache_dir=tmp_path).label("same text")

    assert client_b.calls == 0, "cache on disk should survive a new labeler"


def test_different_models_dont_share_cache(tmp_path):
    """A label produced by model A should not be served when model B asks for it."""
    client_a = FakeClient(model="model-a")
    UrgencyLabeler(client=client_a, cache_dir=tmp_path).label("same text")

    client_b = FakeClient(model="model-b")
    UrgencyLabeler(client=client_b, cache_dir=tmp_path).label("same text")

    assert client_b.calls == 1, "different model means different cache key"


# ----------------------------------------------------------------------------
# Category labeler
# ----------------------------------------------------------------------------


def test_category_labels_ticket_with_valid_json(tmp_path):
    client = FakeClient(response='{"category": "bug", "reasoning": "app crashes"}')
    labeler = CategoryLabeler(client=client, cache_dir=tmp_path)

    label = labeler.label("the app crashes on launch")

    assert label.category == Category.BUG
    assert label.reasoning == "app crashes"
    assert label.rubric_version == CATEGORY_RUBRIC_VERSION
    assert label.model == "fake-model-v1"
    assert client.calls == 1


def test_category_cache_skips_second_call(tmp_path):
    client = FakeClient(response='{"category": "billing", "reasoning": "refund"}')
    labeler = CategoryLabeler(client=client, cache_dir=tmp_path)

    labeler.label("I want a refund")
    labeler.label("I want a refund")

    assert client.calls == 1


def test_category_strips_markdown_code_fences(tmp_path):
    client = FakeClient(
        response='```json\n{"category": "feature_request", "reasoning": "wants dark mode"}\n```'
    )
    labeler = CategoryLabeler(client=client, cache_dir=tmp_path)

    label = labeler.label("please add dark mode")

    assert label.category == Category.FEATURE_REQUEST


def test_category_and_urgency_share_cache_dir_without_collision(tmp_path):
    """Both labelers can write to the same cache directory.

    They differ in rubric_version, so the cache key (sha256 of model+version+text)
    can never collide. Regression test for an earlier version that used the same
    "v1" rubric_version for both — which would have caused one to read the other's
    cached payload and fail with KeyError.
    """
    urgency_client = FakeClient(response='{"urgency": "high", "reasoning": "locked"}')
    category_client = FakeClient(
        response='{"category": "account", "reasoning": "login issue"}'
    )

    text = "I can't log in"
    UrgencyLabeler(client=urgency_client, cache_dir=tmp_path).label(text)
    label = CategoryLabeler(client=category_client, cache_dir=tmp_path).label(text)

    # If the keys collided, this would have loaded the urgency cache file and
    # raised KeyError("category"). It works because rubric_version differs.
    assert label.category == Category.ACCOUNT
    assert category_client.calls == 1, "category labeler must NOT hit urgency's cache"


def test_category_invalid_value_raises(tmp_path):
    """LLM returns a category outside our enum — must blow up, not silently coerce."""
    client = FakeClient(response='{"category": "complaint", "reasoning": "..."}')
    labeler = CategoryLabeler(client=client, cache_dir=tmp_path)

    with pytest.raises(ValueError):
        labeler.label("...")


def test_category_cache_persists_across_instances(tmp_path):
    client_a = FakeClient(response='{"category": "billing", "reasoning": "x"}')
    CategoryLabeler(client=client_a, cache_dir=tmp_path).label("same text")

    client_b = FakeClient(response='{"category": "billing", "reasoning": "x"}')
    CategoryLabeler(client=client_b, cache_dir=tmp_path).label("same text")

    assert client_b.calls == 0
