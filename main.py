from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import Base, engine, get_db
from app.models import Proposal
from app.schemas import ProposalCreate, ProposalResponse

app = FastAPI(title="Simple Proposal API", version="1.0.0")

Base.metadata.create_all(bind=engine)


@app.get("/")
def serve_form():
    frontend_path = Path(__file__).resolve().parents[1] /"PR_ARAM"/ "frontend" / "index.html"
    return FileResponse(frontend_path)


@app.post("/proposals", response_model=ProposalResponse)
def create_proposal(payload: ProposalCreate, db: Session = Depends(get_db)):
    existing = db.query(Proposal).filter(Proposal.id == payload.id).first()
    if existing:
        raise HTTPException(status_code=400, detail="ID already exists.")

    proposal = Proposal(**payload.model_dump())
    db.add(proposal)
    db.commit()
    db.refresh(proposal)
    return proposal


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
