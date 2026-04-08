from datetime import datetime
from enum import StrEnum

from sqlalchemy import DateTime, Enum, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class WorkflowStage(StrEnum):
    REQUESTED = "requested"
    REVIEW = "review"
    DRAFT = "draft"
    APPROVAL = "approval"
    ORDERED = "ordered"


class ProposalRequest(Base):
    __tablename__ = "proposal_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    client_name: Mapped[str] = mapped_column(String(255), nullable=False)
    project_title: Mapped[str] = mapped_column(String(255), nullable=False)
    requirements_summary: Mapped[str] = mapped_column(Text, nullable=False)
    budget: Mapped[float | None] = mapped_column(Float, nullable=True)
    due_date: Mapped[str | None] = mapped_column(String(10), nullable=True)
    stage: Mapped[WorkflowStage] = mapped_column(
        Enum(WorkflowStage), default=WorkflowStage.REQUESTED, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )
