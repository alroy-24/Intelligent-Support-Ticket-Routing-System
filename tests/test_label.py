"""Tests for the urgency labeler. We mock the LLM client entirely — no API calls."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from ticketrouting.data.label import RUBRIC_VERSION, UrgencyLabeler
from ticketrouting.schemas import Urgency


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
