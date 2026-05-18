"""LLM-based synthetic labeling for ticket urgency AND category.

Both labelers sit on top of `ticketrouting.llm.CachedJsonLabeler`, which gives
them temperature=0 JSON-mode calls with sha256-keyed disk caching and tenacity
retries. The only per-labeler code here is the rubric, the cache namespace, and
the parser that maps the JSON payload to a typed dataclass.

Why two rubrics:

- Urgency is the noisiest label (human triagers disagree ~25% on High vs
  Critical) so the rubric spends tokens disambiguating adjacent levels and
  explicitly says "when in doubt pick the lower one".
- Category is less ambiguous individually but has the trickier failure mode of
  the LLM defaulting to OTHER for anything it isn't sure about. The rubric
  pushes back by giving sharp boundaries for technical vs bug vs account.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Re-exported for back-compat with callers (scripts/build_dataset.py imports
# make_client from here).
from ticketrouting.llm import (  # noqa: F401
    AnthropicClient,
    CachedJsonLabeler,
    GroqClient,
    LLMClient,
    make_client,
    strip_code_fences,
)
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
# Urgency labeler
# ----------------------------------------------------------------------------


@dataclass
class UrgencyLabel:
    urgency: Urgency
    reasoning: str
    model: str
    rubric_version: str


class UrgencyLabeler(CachedJsonLabeler):
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
        data = json.loads(strip_code_fences(raw))
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


class CategoryLabeler(CachedJsonLabeler):
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
        data = json.loads(strip_code_fences(raw))
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
