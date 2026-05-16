"""LLM-based synthetic labeling for ticket urgency.

We use a chat-completion LLM to label urgency because the Bitext dataset has no urgency
annotation and hand-labeling 20k tickets is impractical. The labeler is designed for:

- **Determinism:** temperature=0 + sha256-keyed disk cache so re-running gives identical
  labels and re-labels nothing.
- **Robustness:** retries with exponential backoff; the model is asked to return strict
  JSON and we validate before accepting.
- **Provider-agnostic:** swap Anthropic for Groq for local Ollama with no labeler changes;
  the labeler only talks to the `LLMClient` protocol.
- **Auditability:** every labeled row carries the model id and the rubric version, so
  when we re-prompt later we can tell which labels are stale.

The rubric is deliberately verbose. Urgency is the noisiest label in the dataset
(human triagers disagree ~25% of the time on High vs. Critical) so we spend the
tokens to anchor the model.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tenacity import retry, stop_after_attempt, wait_exponential

from ticketrouting.schemas import Urgency

RUBRIC_VERSION = "v1"

URGENCY_RUBRIC = """\
You are labeling customer support tickets for urgency. Read the ticket and assign EXACTLY ONE level.

Levels:
- "critical": Production outage, data loss, security breach, payment failures blocking business,
  enterprise customer threatening churn, regulated-deadline failure (tax, legal, compliance).
- "high": Account is locked or inaccessible; recurring/duplicate charges; broken feature blocking
  the customer's main workflow; angry sentiment from a paying customer; clear deadline within 24h.
- "medium": Single broken feature with a workaround; billing question without active financial loss;
  account change request; mild frustration; reply needed within a few days.
- "low": General inquiry, feature request, documentation question, satisfaction note, marketing
  unsubscribe, low-stakes how-to question.

Rules:
1. When uncertain between two adjacent levels, pick the LOWER one. Over-escalation costs trust.
2. Sentiment alone does not raise urgency — an angry user with a low-stakes issue is still "medium".
3. A mention of "ASAP" or "urgent" by the customer does NOT automatically mean critical. Judge the
   underlying issue, not the customer's framing.
4. If the ticket is ambiguous or empty, return "low".

Output STRICT JSON only, no prose:
{"urgency": "<low|medium|high|critical>", "reasoning": "<one short sentence>"}
"""


# ----------------------------------------------------------------------------
# Client protocol + implementations
# ----------------------------------------------------------------------------


class LLMClient(Protocol):
    """Minimal protocol so tests can swap in a fake and we can swap providers."""

    model: str

    def complete(self, system: str, user: str) -> str: ...


class AnthropicClient:
    """Wraps the Anthropic SDK. Lazy-imports so tests don't need the dep."""

    DEFAULT_MODEL = "claude-haiku-4-5-20251001"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        from anthropic import Anthropic

        self.model = model or self.DEFAULT_MODEL
        self._client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def complete(self, system: str, user: str) -> str:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=200,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text


class GroqClient:
    """Wraps the Groq SDK (OpenAI-compatible). Free tier, fast Llama inference."""

    DEFAULT_MODEL = "llama-3.1-8b-instant"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        from groq import Groq

        self.model = model or self.DEFAULT_MODEL
        self._client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=200,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


def make_client(provider: str = "auto", model: str | None = None) -> LLMClient:
    """Build an LLM client. `provider` is one of: auto, groq, anthropic.

    `auto` picks based on which API key is set (Groq preferred since it's free).
    """
    if provider == "auto":
        if os.environ.get("GROQ_API_KEY"):
            provider = "groq"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        else:
            raise RuntimeError(
                "No LLM API key found. Set GROQ_API_KEY or ANTHROPIC_API_KEY in your .env."
            )

    if provider == "groq":
        return GroqClient(model=model)
    if provider == "anthropic":
        return AnthropicClient(model=model)
    raise ValueError(f"Unknown provider: {provider!r}")


# ----------------------------------------------------------------------------
# Labeler
# ----------------------------------------------------------------------------


@dataclass
class UrgencyLabel:
    urgency: Urgency
    reasoning: str
    model: str
    rubric_version: str


class UrgencyLabeler:
    """Label tickets for urgency with disk-cached, retried LLM calls."""

    def __init__(
        self,
        client: LLMClient | None = None,
        cache_dir: Path | str | None = None,
    ):
        self._client = client or make_client()
        self._cache_dir = Path(cache_dir) if cache_dir else Path(".cache/llm")
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def label(self, ticket_text: str) -> UrgencyLabel:
        cached = self._cache_get(ticket_text)
        if cached is not None:
            return cached

        raw = self._call_llm(ticket_text)
        label = self._parse(raw)
        self._cache_put(ticket_text, label)
        return label

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
    def _call_llm(self, ticket_text: str) -> str:
        return self._client.complete(
            system=URGENCY_RUBRIC,
            user=f"Ticket:\n{ticket_text.strip()}",
        )

    def _parse(self, raw: str) -> UrgencyLabel:
        # Models sometimes wrap JSON in code fences despite instructions — strip them.
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        data = json.loads(text)
        return UrgencyLabel(
            urgency=Urgency(data["urgency"].lower()),
            reasoning=str(data.get("reasoning", "")).strip(),
            model=self._client.model,
            rubric_version=RUBRIC_VERSION,
        )

    def _cache_key(self, ticket_text: str) -> str:
        h = hashlib.sha256()
        h.update(self._client.model.encode())
        h.update(RUBRIC_VERSION.encode())
        h.update(ticket_text.encode())
        return h.hexdigest()

    def _cache_path(self, ticket_text: str) -> Path:
        return self._cache_dir / f"{self._cache_key(ticket_text)}.json"

    def _cache_get(self, ticket_text: str) -> UrgencyLabel | None:
        p = self._cache_path(ticket_text)
        if not p.exists():
            return None
        data = json.loads(p.read_text())
        return UrgencyLabel(
            urgency=Urgency(data["urgency"]),
            reasoning=data["reasoning"],
            model=data["model"],
            rubric_version=data["rubric_version"],
        )

    def _cache_put(self, ticket_text: str, label: UrgencyLabel) -> None:
        p = self._cache_path(ticket_text)
        p.write_text(
            json.dumps(
                {
                    "urgency": label.urgency.value,
                    "reasoning": label.reasoning,
                    "model": label.model,
                    "rubric_version": label.rubric_version,
                }
            )
        )
