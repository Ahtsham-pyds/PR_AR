from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import ProposalRequest, WorkflowStage
from app.schemas import ProposalCreate, ProposalResponse, StageUpdate

router = APIRouter(prefix="/proposals", tags=["proposals"])

ALLOWED_STAGE_TRANSITIONS: dict[WorkflowStage, set[WorkflowStage]] = {
    WorkflowStage.REQUESTED: {WorkflowStage.REVIEW},
    WorkflowStage.REVIEW: {WorkflowStage.DRAFT},
    WorkflowStage.DRAFT: {WorkflowStage.APPROVAL},
    WorkflowStage.APPROVAL: {WorkflowStage.ORDERED},
    WorkflowStage.ORDERED: set(),
}


@router.post("", response_model=ProposalResponse)
def create_proposal(payload: ProposalCreate, db: Session = Depends(get_db)):
    proposal = ProposalRequest(**payload.model_dump())
    db.add(proposal)
    db.commit()
    db.refresh(proposal)
    return proposal


@router.get("", response_model=list[ProposalResponse])
def list_proposals(db: Session = Depends(get_db)):
    return db.query(ProposalRequest).order_by(ProposalRequest.created_at.desc()).all()


@router.get("/{proposal_id}", response_model=ProposalResponse)
def get_proposal(proposal_id: int, db: Session = Depends(get_db)):
    proposal = db.query(ProposalRequest).filter(ProposalRequest.id == proposal_id).first()
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found.")
    return proposal


@router.patch("/{proposal_id}/stage", response_model=ProposalResponse)
def update_stage(proposal_id: int, payload: StageUpdate, db: Session = Depends(get_db)):
    proposal = db.query(ProposalRequest).filter(ProposalRequest.id == proposal_id).first()
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found.")

    current_stage = proposal.stage
    if payload.stage == current_stage:
        return proposal

    if payload.stage not in ALLOWED_STAGE_TRANSITIONS[current_stage]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid transition from '{current_stage.value}' "
                f"to '{payload.stage.value}'."
            ),
        )

    proposal.stage = payload.stage
    db.add(proposal)
    db.commit()
    db.refresh(proposal)
    return proposal
