from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas import FormSuggestion, PromptInput
from app.services.extraction import suggest_from_text

router = APIRouter(prefix="/intake", tags=["intake"])


@router.post("/prompt", response_model=FormSuggestion)
def analyze_prompt(payload: PromptInput):
    return suggest_from_text(payload.prompt)


@router.post("/file", response_model=FormSuggestion)
async def analyze_file(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    supported_extensions = (".txt", ".md", ".csv")
    if file.filename and not file.filename.lower().endswith(supported_extensions):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use .txt, .md, or .csv for v1.",
        )

    text = raw.decode("utf-8", errors="ignore")
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract readable text.")

    return suggest_from_text(text)
