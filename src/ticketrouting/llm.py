"""Shared LLM machinery: client protocol, providers, and a cached JSON-mode base.

Three users live on top of this module: UrgencyLabeler, CategoryLabeler
(both in `data/label.py`), and TicketSummarizer (`models/summary/summarizer.py`).
The pattern is identical across all three — temperature=0 JSON-mode call with
sha256-keyed disk cache, tenacity retries, and a typed Python output — so the
machinery lives here and each user only contributes a rubric + a parser.

Cache-key design: sha256(model + rubric_version + text). Each user picks a
namespaced rubric_version ("v1" for urgency, "cat-v1" for category, "sum-v1"
for summarizer) so the three can share a cache directory without collisions.
A rubric edit means bumping the version, which transparently invalidates only
the affected entries.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Protocol

from tenacity import retry, stop_after_attempt, wait_exponential


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
            max_tokens=600,
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
            max_tokens=600,
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
# Shared cached JSON-mode base
# ----------------------------------------------------------------------------


def strip_code_fences(text: str) -> str:
    """Models sometimes wrap JSON in ```json``` despite instructions — strip them."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


class CachedJsonLabeler:
    """Base for any temperature=0 JSON-mode LLM call with disk caching.

    Subclasses set `rubric` (system prompt) and `rubric_version` (cache
    namespace), implement `_parse(raw)` to return a typed result, and provide
    `_cache_get` / `_cache_put` for their specific payload shape.
    """

    rubric: str = ""
    rubric_version: str = ""

    def __init__(
        self,
        client: LLMClient | None = None,
        cache_dir: Path | str | None = None,
    ):
        self._client = client or make_client()
        self._cache_dir = Path(cache_dir) if cache_dir else Path(".cache/llm")
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
    def _call_llm(self, ticket_text: str) -> str:
        return self._client.complete(
            system=self.rubric,
            user=f"Ticket:\n{ticket_text.strip()}",
        )

    def _cache_key(self, ticket_text: str) -> str:
        h = hashlib.sha256()
        h.update(self._client.model.encode())
        h.update(self.rubric_version.encode())
        h.update(ticket_text.encode())
        return h.hexdigest()

    def _cache_path(self, ticket_text: str) -> Path:
        return self._cache_dir / f"{self._cache_key(ticket_text)}.json"
