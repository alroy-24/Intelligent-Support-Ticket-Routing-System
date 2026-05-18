"""FastAPI app for the routing service.

Run locally:
  uvicorn ticketrouting.api.app:app --reload

Then:
  curl http://localhost:8000/health
  curl -X POST http://localhost:8000/route \
       -H "content-type: application/json" \
       -d '{"text": "We were charged twice this month, please refund"}'

The /health endpoint reports which models are loaded so the caller can
distinguish a real prediction from a degraded fallback. /route always
returns 200 with a RoutingDecision — missing/failing models produce a
typed fallback rather than a 500.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI

from ticketrouting.api.service import RoutingService
from ticketrouting.schemas import RoutingDecision, TicketIn

load_dotenv()

_service: RoutingService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service
    _service = RoutingService.from_env()
    yield
    _service = None


app = FastAPI(
    title="Ticket Routing API",
    description="Routes support tickets to teams, scores urgency, and summarises.",
    version="0.1.0",
    lifespan=lifespan,
)


def get_service() -> RoutingService:
    """FastAPI dependency. Tests override this to inject a fake service."""
    if _service is None:
        # Fall back to from_env so tests that don't trigger lifespan still work.
        return RoutingService.from_env()
    return _service


@app.get("/health")
def health(service: RoutingService = Depends(get_service)) -> dict:
    """Liveness + which models are actually loaded."""
    loaded = service.loaded
    return {
        "status": "ok",
        "models_loaded": loaded,
        "fully_loaded": all(loaded.values()),
    }


@app.post("/route", response_model=RoutingDecision)
async def route(
    ticket: TicketIn, service: RoutingService = Depends(get_service)
) -> RoutingDecision:
    return await service.route(ticket)
