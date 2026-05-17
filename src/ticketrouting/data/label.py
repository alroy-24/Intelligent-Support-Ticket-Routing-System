"""LLM-based synthetic labeling for ticket urgency AND category.

Both labelers share the same skeleton: a temperature-0 JSON-mode LLM call,
sha256-keyed disk cache (so reruns are free), tenacity-backed retries, and a
rubric that bakes in the failure modes we've observed. The differences live in
the rubric, the JSON keys, and the typed output dataclass.

Design choices:

- **Determinism:** temperature=0 + cache key includes model + rubric version,
  so a rubric tweak invalidates only the affected labels (not the entire cache).
- **Auditability:** every labeled row carries the model id and the rubric
  version, so when we re-prompt later we can tell which labels are stale.
- **Provider-agnostic:** swap Anthropic for Groq with no labeler changes; the
  labeler only talks to the `LLMClient` protocol.

Why two rubrics:

- Urgency is the noisiest label (human triagers disagree ~25% on High vs
  Critical) so the rubric spends tokens disambiguating adjacent levels and
  explicitly says "when in doubt pick the lower one".
- Category is less ambiguous individually but has the trickier failure mode of
  the LLM defaulting to OTHER for anything it isn't sure about. The rubric
  pushes back by giving sharp boundaries for technical vs bug vs account.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tenacity import retry, stop_after_attempt, wait_exponential

from ticketrouting.schemas import Category, Urgency

RUBRIC_VERSION = "v1"
# Distinct from urgency's "v1" so the two labelers can't collide on the same
# (model, version, text) cache key.
CATEGORY_RUBRIC_VERSION = "cat-v1"

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

CATEGORY_RUBRIC = """\
You are labeling customer support tickets for category. Read the ticket and assign EXACTLY ONE.

Categories:
- "billing": Payments, charges, refunds, invoices, subscription or plan changes, order status,
  pricing questions, anything involving money flowing between customer and company.
- "technical": System-level engineering or infrastructure problems — the site/app/API is down,
  login service is failing for many users, severe performance degradation, integration or webhook
  failures, certificate or DNS issues. The hallmark is "this affects many users or core systems",
  not "one button is broken for me".
- "account": A single user's profile, login, password, registration, identity, verification, or
  account-lifecycle issue. If the user simply cannot access THEIR account (not a system outage),
  this is account, not technical.
- "bug": A specific feature behaving incorrectly for an individual user — UI glitch, wrong
  calculation, crash on one specific action, button doesn't respond, display issue. One feature,
  not the whole product.
- "feature_request": The customer is suggesting a NEW capability, enhancement, or improvement that
  the product doesn't currently have. Phrases like "it would be great if...", "can you add...",
  "I wish...".
- "other": General inquiry, thanks/praise, complaint without an actionable issue, shipping or
  logistics, marketing/unsubscribe, anything that doesn't clearly fit above.

Rules:
1. Pick EXACTLY ONE — the category the customer is most clearly asking about.
2. Distinguish carefully: "technical" = system-wide; "bug" = one feature misbehaving; "account" =
   one user's identity/login problem.
3. Pure complaints or feedback with no specific actionable issue go to "other", NOT "bug".
4. If the ticket is ambiguous, empty, or just a greeting, return "other".

Output STRICT JSON only, no prose:
{"category": "<billing|technical|account|bug|feature_request|other>", "reasoning": "<one short sentence>"}
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
# Shared cached-LLM-call machinery
# ----------------------------------------------------------------------------


def _strip_code_fences(text: str) -> str:
    """Models sometimes wrap JSON in ```json``` despite instructions — strip them."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


class _CachedJsonLabeler:
    """Shared base: cache key by (model, rubric_version, text); retry; JSON parse.

    Subclasses set `rubric` (system prompt) and `rubric_version`, then implement
    `_parse_payload(data)` which receives the already-decoded JSON dict and
    returns a typed label.
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


# ----------------------------------------------------------------------------
# Urgency labeler
# ----------------------------------------------------------------------------


@dataclass
class UrgencyLabel:
    urgency: Urgency
    reasoning: str
    model: str
    rubric_version: str


class UrgencyLabeler(_CachedJsonLabeler):
    """Label tickets for urgency with disk-cached, retried LLM calls."""

    rubric = URGENCY_RUBRIC
    rubric_version = RUBRIC_VERSION

    def label(self, ticket_text: str) -> UrgencyLabel:
        cached = self._cache_get(ticket_text)
        if cached is not None:
            return cached

        raw = self._call_llm(ticket_text)
        label = self._parse(raw)
        self._cache_put(ticket_text, label)
        return label

    def _parse(self, raw: str) -> UrgencyLabel:
        data = json.loads(_strip_code_fences(raw))
        return UrgencyLabel(
            urgency=Urgency(data["urgency"].lower()),
            reasoning=str(data.get("reasoning", "")).strip(),
            model=self._client.model,
            rubric_version=self.rubric_version,
        )

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


# ----------------------------------------------------------------------------
# Category labeler
# ----------------------------------------------------------------------------


@dataclass
class CategoryLabel:
    category: Category
    reasoning: str
    model: str
    rubric_version: str


class CategoryLabeler(_CachedJsonLabeler):
    """Label tickets for category with disk-cached, retried LLM calls.

    Built specifically for Twitter ingestion: Bitext's intent column gives us
    BILLING / ACCOUNT / OTHER for free, but Twitter has no labels at all and
    is the only source for BUG / TECHNICAL / FEATURE_REQUEST examples.
    """

    rubric = CATEGORY_RUBRIC
    rubric_version = CATEGORY_RUBRIC_VERSION

    def label(self, ticket_text: str) -> CategoryLabel:
        cached = self._cache_get(ticket_text)
        if cached is not None:
            return cached

        raw = self._call_llm(ticket_text)
        label = self._parse(raw)
        self._cache_put(ticket_text, label)
        return label

    def _parse(self, raw: str) -> CategoryLabel:
        data = json.loads(_strip_code_fences(raw))
        return CategoryLabel(
            category=Category(data["category"].lower()),
            reasoning=str(data.get("reasoning", "")).strip(),
            model=self._client.model,
            rubric_version=self.rubric_version,
        )

    def _cache_get(self, ticket_text: str) -> CategoryLabel | None:
        p = self._cache_path(ticket_text)
        if not p.exists():
            return None
        data = json.loads(p.read_text())
        return CategoryLabel(
            category=Category(data["category"]),
            reasoning=data["reasoning"],
            model=data["model"],
            rubric_version=data["rubric_version"],
        )

    def _cache_put(self, ticket_text: str, label: CategoryLabel) -> None:
        p = self._cache_path(ticket_text)
        p.write_text(
            json.dumps(
                {
                    "category": label.category.value,
                    "reasoning": label.reasoning,
                    "model": label.model,
                    "rubric_version": label.rubric_version,
                }
            )
        )
