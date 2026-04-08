from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models import WorkflowStage


class FormSuggestion(BaseModel):
    client_name: str | None = None
    project_title: str | None = None
    requirements_summary: str | None = None
    budget: float | None = None
    due_date: str | None = Field(default=None, description="YYYY-MM-DD")
    confidence: float = 0.0
    notes: list[str] = Field(default_factory=list)


class PromptInput(BaseModel):
    prompt: str = Field(min_length=10)


class ProposalCreate(BaseModel):
    client_name: str
    project_title: str
    requirements_summary: str
    budget: float | None = None
    due_date: str | None = None


class StageUpdate(BaseModel):
    stage: WorkflowStage


class ProposalResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    client_name: str
    project_title: str
    requirements_summary: str
    budget: float | None
    due_date: str | None
    stage: WorkflowStage
    created_at: datetime
    updated_at: datetime
