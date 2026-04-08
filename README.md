# Proposal Workflow Backend (Python)

FastAPI backend for a proposal lifecycle:

1. AI-assisted intake from prompt or file
2. Proposal request creation
3. Stage progression: `requested -> review -> draft -> approval -> ordered`

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open:

- Swagger UI: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

## Key endpoints

- `POST /intake/prompt`  
  Accepts free-text prompt and returns suggested form fields.
- `POST /intake/file`  
  Accepts `.txt`, `.md`, `.csv` upload and returns suggested form fields.
- `POST /proposals`  
  Creates a proposal request record.
- `GET /proposals`  
  Lists records.
- `PATCH /proposals/{id}/stage`  
  Moves workflow to next allowed stage.

## Notes

- v1 extraction is rule-based for reliability and quick start.
- You can later swap extraction service with OpenAI responses while keeping API shape stable.
