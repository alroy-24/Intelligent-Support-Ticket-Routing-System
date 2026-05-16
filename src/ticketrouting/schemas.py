"""Shared pydantic schemas — the API contract for tickets and routing decisions."""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Category(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    BUG = "bug"
    FEATURE_REQUEST = "feature_request"
    OTHER = "other"


class Urgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"


class TicketIn(BaseModel):
    ticket_id: str | None = None
    text: str = Field(..., min_length=1)
    customer_tier: Literal["free", "pro", "scale", "enterprise"] | None = None
    locale: str | None = None


class Entities(BaseModel):
    product: str | None = None
    error_codes: list[str] = Field(default_factory=list)
    sentiment: Sentiment | None = None
    deadline_mentioned: str | None = None


class RoutingDecision(BaseModel):
    route_to: list[Category]
    route_confidence: float = Field(..., ge=0.0, le=1.0)
    urgency: Urgency
    urgency_score: float = Field(..., ge=0.0, le=1.0)
    summary: str | None = None
    entities: Entities | None = None
    latency_ms: int | None = None
