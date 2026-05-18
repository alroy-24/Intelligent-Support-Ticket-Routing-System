"""RoutingService — orchestrates the three models into one RoutingDecision.

Design follows the README's `## 2. System design` section:
- Category and Urgency are fast classical classifiers, called in parallel.
- Summarizer is a slow LLM call with its own time budget; if it misses the
  deadline (or the LLM is rate-limited / unreachable), the response is still
  returned with summary=None and entities=None. Per the README, this is the
  point where a follow-up webhook would patch the summary in — out of scope
  for v1 but the deadline contract makes that extension straightforward.

Graceful degradation is a deliberate v1 choice: any of the three components
can be missing (artifact not yet trained, no LLM key, rate-limited) and the
service still responds. The /health endpoint reports which are live so the
caller can tell the difference between "model said OTHER" and "no model".
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

from ticketrouting.active.loop import LogRow, PredictionLog
from ticketrouting.models.category.baseline import load as load_category_pipe
from ticketrouting.models.category.baseline import predict_one as predict_category
from ticketrouting.models.summary.summarizer import TicketSummarizer, TicketSummary
from ticketrouting.models.urgency.baseline import load as load_urgency_pipe
from ticketrouting.models.urgency.baseline import predict_one as predict_urgency
from ticketrouting.schemas import Category, RoutingDecision, TicketIn, Urgency

DEFAULT_CATEGORY_PATH = Path("artifacts/baseline_category.joblib")
DEFAULT_URGENCY_PATH = Path("artifacts/urgency_baseline.joblib")
DEFAULT_SUMMARY_DEADLINE_S = 8.0
ACTIVE_LOG_ENV_VAR = "ACTIVE_LOG_PATH"


class RoutingService:
    """End-to-end orchestrator. Loads models once, serves many requests.

    All three predictions run concurrently via asyncio.to_thread so that a
    single ticket's wall-clock is bounded by the slowest model (the LLM) and
    the two classical ones come "for free" in parallel.
    """

    def __init__(
        self,
        category_pipe=None,
        urgency_pipe=None,
        summarizer: TicketSummarizer | None = None,
        summary_deadline_s: float = DEFAULT_SUMMARY_DEADLINE_S,
        prediction_log: PredictionLog | None = None,
    ):
        self.category_pipe = category_pipe
        self.urgency_pipe = urgency_pipe
        self.summarizer = summarizer
        self.summary_deadline_s = summary_deadline_s
        self.prediction_log = prediction_log

    @classmethod
    def from_env(
        cls,
        category_path: Path | None = None,
        urgency_path: Path | None = None,
        enable_summarizer: bool | None = None,
        summary_deadline_s: float = DEFAULT_SUMMARY_DEADLINE_S,
        prediction_log_path: Path | None = None,
    ) -> "RoutingService":
        """Load whatever's available on disk + reachable. Never raises on missing pieces."""
        category_pipe = _try_load(category_path or DEFAULT_CATEGORY_PATH, load_category_pipe)
        urgency_pipe = _try_load(urgency_path or DEFAULT_URGENCY_PATH, load_urgency_pipe)

        if enable_summarizer is None:
            enable_summarizer = bool(
                os.environ.get("GROQ_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
            )
        summarizer = TicketSummarizer() if enable_summarizer else None

        # Active-learning log: opt-in via env var so tests don't accidentally write files.
        if prediction_log_path is None:
            env_path = os.environ.get(ACTIVE_LOG_ENV_VAR)
            prediction_log_path = Path(env_path) if env_path else None
        prediction_log = PredictionLog(prediction_log_path) if prediction_log_path else None

        return cls(
            category_pipe=category_pipe,
            urgency_pipe=urgency_pipe,
            summarizer=summarizer,
            summary_deadline_s=summary_deadline_s,
            prediction_log=prediction_log,
        )

    @property
    def loaded(self) -> dict[str, bool]:
        return {
            "category": self.category_pipe is not None,
            "urgency": self.urgency_pipe is not None,
            "summarizer": self.summarizer is not None,
        }

    async def route(self, ticket: TicketIn) -> RoutingDecision:
        start = time.perf_counter()

        cat_task = asyncio.create_task(asyncio.to_thread(self._safe_category, ticket.text))
        urg_task = asyncio.create_task(asyncio.to_thread(self._safe_urgency, ticket.text))
        sum_task = asyncio.create_task(asyncio.to_thread(self._safe_summarize, ticket.text))

        # Category + urgency are fast (ms); we always wait for them.
        category, urgency = await asyncio.gather(cat_task, urg_task)

        # Summary has its own budget — if it overruns we drop it.
        try:
            summary_result = await asyncio.wait_for(sum_task, timeout=self.summary_deadline_s)
        except asyncio.TimeoutError:
            sum_task.cancel()
            summary_result = None

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        decision = RoutingDecision(
            route_to=[category[0]],
            route_confidence=category[1],
            urgency=urgency[0],
            urgency_score=urgency[1],
            summary=summary_result.summary if summary_result else None,
            entities=summary_result.entities if summary_result else None,
            latency_ms=elapsed_ms,
        )

        # Active-learning log — never blocks or fails the response.
        if self.prediction_log is not None:
            try:
                self.prediction_log.append(
                    LogRow(
                        text=ticket.text,
                        predicted_category=decision.route_to[0].value,
                        route_confidence=decision.route_confidence,
                        predicted_urgency=decision.urgency.value,
                        urgency_score=decision.urgency_score,
                        ticket_id=ticket.ticket_id,
                    )
                )
            except Exception:
                pass

        return decision

    # ------------------------------------------------------------------------
    # Per-model wrappers — each catches its own exceptions so a single failure
    # never tanks the request.
    # ------------------------------------------------------------------------

    def _safe_category(self, text: str) -> tuple[Category, float]:
        if self.category_pipe is None:
            return (Category.OTHER, 0.0)
        try:
            return predict_category(self.category_pipe, text)
        except Exception:
            return (Category.OTHER, 0.0)

    def _safe_urgency(self, text: str) -> tuple[Urgency, float]:
        if self.urgency_pipe is None:
            return (Urgency.LOW, 0.0)
        try:
            return predict_urgency(self.urgency_pipe, text)
        except Exception:
            return (Urgency.LOW, 0.0)

    def _safe_summarize(self, text: str) -> TicketSummary | None:
        if self.summarizer is None:
            return None
        try:
            return self.summarizer.summarize(text)
        except Exception:
            return None


def _try_load(path: Path, loader):
    if not path.exists():
        return None
    try:
        return loader(path)
    except Exception:
        return None
