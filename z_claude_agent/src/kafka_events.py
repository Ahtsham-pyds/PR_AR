"""
SOW Agentic Memory System — message schema contracts (Phase 1)

These Pydantic models are the data contract at every boundary in the
ingestion pipeline. Nothing gets published to Kafka, and nothing gets
written to Neo4j/Postgres, without passing through one of these models
first. This is deliberate: a schema-validated boundary is what lets a
Kafka DLQ do its job (bad data gets caught HERE, not three steps later).
"""

from __future__ import annotations
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional
from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Inbound events — produced by the frontend form or the chat interface,
# published to Kafka topics `sow.form.events` / `sow.chat.events`.
# ---------------------------------------------------------------------------

class BaseEvent(BaseModel):
    event_id: str = Field(default_factory=_new_id)
    session_id: str
    project_id: str
    user_id: str
    timestamp: datetime = Field(default_factory=_now)


class FormEvent(BaseEvent):
    """A structured submission from the frontend SOW form."""
    channel: Literal["form"] = "form"
    form_type: Literal[
        "scope_item", "budget_line", "standard_reference", "stakeholder_assignment"
    ]
    payload: dict


class ChatEvent(BaseEvent):
    """One turn of the chat interface."""
    channel: Literal["chat"] = "chat"
    message_role: Literal["user", "agent"]
    message_text: str


# ---------------------------------------------------------------------------
# Extraction output — produced by extraction workers (Phase 3),
# published to `sow.facts.extracted`. NOT yet trusted; every fact starts
# in `status="pending"` and must clear the review queue before it is
# allowed to update the Neo4j graph.
# ---------------------------------------------------------------------------

class ExtractedFact(BaseModel):
    fact_id: str = Field(default_factory=_new_id)
    source_event_id: str
    session_id: str
    project_id: str
    fact_text: str
    entity_type: Literal[
        "scope_item", "budget_line", "standard_reference", "stakeholder", "other"
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    extracted_at: datetime = Field(default_factory=_now)
    status: Literal["pending", "approved", "rejected", "needs_edit"] = "pending"


# ---------------------------------------------------------------------------
# Review decisions — produced by a human reviewer action in the UI,
# published to `sow.review.decisions`. Consumed by the memory-writer
# (Phase 4) to actually commit approved facts into Neo4j.
# ---------------------------------------------------------------------------

class ReviewDecision(BaseModel):
    fact_id: str
    reviewer_id: str
    decision: Literal["approved", "rejected", "needs_edit"]
    edited_text: Optional[str] = None
    reviewed_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Dead letter envelope — anything that fails schema validation or
# processing at any stage gets wrapped in this and published to the
# topic's `.dlq` counterpart (e.g. `sow.form.events.dlq`).
# ---------------------------------------------------------------------------

class DLQEnvelope(BaseModel):
    original_topic: str
    original_payload: dict
    error_message: str
    retry_count: int = 0
    failed_at: datetime = Field(default_factory=_now)