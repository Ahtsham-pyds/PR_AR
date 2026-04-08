from fastapi import FastAPI

from app.database import Base, engine
from app.routes.intake import router as intake_router
from app.routes.proposals import router as proposals_router

app = FastAPI(
    title="Proposal Request to Order API",
    description=(
        "Backend for AI-assisted proposal intake and workflow progression "
        "from request to proposal order."
    ),
    version="0.1.0",
)

Base.metadata.create_all(bind=engine)

app.include_router(intake_router)
app.include_router(proposals_router)


@app.get("/health")
def health():
    return {"status": "ok"}
