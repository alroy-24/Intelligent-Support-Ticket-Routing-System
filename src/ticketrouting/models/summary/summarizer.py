"""LLM-based ticket summariser + entity extractor — the third of the three models.

This is the one place in the system where an LLM is clearly worth the latency
and cost: generative summarisation + structured entity extraction would require
their own labelled corpora to train a small model on, and the output quality of
a fine-tuned-small-model approach lags an off-the-shelf LLM by a wide margin
for the engineering hours involved.

Sitting on top of `ticketrouting.llm.CachedJsonLabeler` for the same reasons
the urgency/category labelers do:
- Determinism (temperature=0).
- sha256-keyed disk cache so repeated tickets (very common in support — same
  template complaints) cost nothing on the second call. The README's cost-
  control section is built on this assumption.
- Auditability — every cached payload records the model id and rubric version
  so we can identify stale summaries when we re-prompt later.

The rubric explicitly forbids invention. The most common LLM failure mode here
is confidently filling in `product` or `deadline_mentioned` when the ticket
doesn't actually contain that information. The structured-output contract
(pydantic Entities) catches schema violations; the rubric handles semantic
fidelity.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ticketrouting.llm import CachedJsonLabeler, strip_code_fences
from ticketrouting.schemas import Entities, Sentiment

SUMMARY_RUBRIC_VERSION = "sum-v1"

SUMMARY_RUBRIC = """\
You are summarising a customer support ticket for the agent who will handle it.

For the ticket below, return:
1. A 1-2 sentence summary capturing the customer's ACTUAL problem and any
   deadline or business context. Write it as the agent's quick-take, not a
   paraphrase of the customer's exact words.
2. Structured entities:
   - "product": the product, feature, plan, or service named in the ticket,
     or null if not named.
   - "error_codes": any error codes mentioned (alphanumeric identifiers, HTTP
     status codes, named exceptions), as a list of strings. Empty list if none.
   - "sentiment": EXACTLY one of "positive", "neutral", "frustrated", "angry"
     — judge from tone and word choice, not from problem severity.
   - "deadline_mentioned": any explicit time-bound context the customer
     raised (e.g. "before end of month", "by Friday"), or null.

Rules:
1. Do NOT invent details that aren't in the ticket. If a field has no evidence,
   use null (or [] for error_codes). Hallucinated entities are worse than empty
   ones because the agent will act on them.
2. "sentiment" describes the customer's emotional state, NOT the severity of
   the underlying issue. A calm enterprise customer reporting an outage is
   "neutral", not "angry".
3. Keep the summary tight. The agent already sees the original ticket — your
   job is to compress, not to repeat.
4. If the ticket is empty or unintelligible, summary = "Unintelligible ticket."
   and every entity field is null/empty.

Output STRICT JSON only, no prose:
{
  "summary": "<1-2 sentences>",
  "entities": {
    "product": "<string>" or null,
    "error_codes": ["<string>", ...],
    "sentiment": "positive" | "neutral" | "frustrated" | "angry",
    "deadline_mentioned": "<string>" or null
  }
}
"""


@dataclass
class TicketSummary:
    summary: str
    entities: Entities
    model: str
    rubric_version: str


class TicketSummarizer(CachedJsonLabeler):
    """Summarise a ticket and extract structured entities in one LLM call."""

    rubric = SUMMARY_RUBRIC
    rubric_version = SUMMARY_RUBRIC_VERSION

    def summarize(self, ticket_text: str) -> TicketSummary:
        cached = self._cache_get(ticket_text)
        if cached is not None:
            return cached

        raw = self._call_llm(ticket_text)
        result = self._parse(raw)
        self._cache_put(ticket_text, result)
        return result

    def _parse(self, raw: str) -> TicketSummary:
        data = json.loads(strip_code_fences(raw))
        entities_payload = dict(data.get("entities") or {})
        # Normalise sentiment case so the LLM saying "Frustrated" still validates.
        if entities_payload.get("sentiment"):
            entities_payload["sentiment"] = str(entities_payload["sentiment"]).lower()
        # Coerce a None error_codes back to [] — the schema requires a list.
        if entities_payload.get("error_codes") is None:
            entities_payload["error_codes"] = []
        entities = Entities(**entities_payload)
        return TicketSummary(
            summary=str(data["summary"]).strip(),
            entities=entities,
            model=self._client.model,
            rubric_version=self.rubric_version,
        )

    def _cache_get(self, ticket_text: str) -> TicketSummary | None:
        p = self._cache_path(ticket_text)
        if not p.exists():
            return None
        data = json.loads(p.read_text())
        return TicketSummary(
            summary=data["summary"],
            entities=Entities(**data["entities"]),
            model=data["model"],
            rubric_version=data["rubric_version"],
        )

    def _cache_put(self, ticket_text: str, result: TicketSummary) -> None:
        p = self._cache_path(ticket_text)
        p.write_text(
            json.dumps(
                {
                    "summary": result.summary,
                    "entities": _entities_to_payload(result.entities),
                    "model": result.model,
                    "rubric_version": result.rubric_version,
                }
            )
        )


def _entities_to_payload(e: Entities) -> dict:
    return {
        "product": e.product,
        "error_codes": list(e.error_codes),
        "sentiment": e.sentiment.value if isinstance(e.sentiment, Sentiment) else e.sentiment,
        "deadline_mentioned": e.deadline_mentioned,
    }
