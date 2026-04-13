# SOW Forge — AI-Powered Scope of Work Builder

A full-stack SOW builder for Saudi Arabia energy/construction contracts, with intelligent document parsing, persistent RAG memory, and professional PDF generation.

---

## Tabs

| Tab | Section | Contents |
|-----|---------|---------|
| 1 | Requirements | Upload requirement doc → LlamaParse → GPT-4o fills all tabs |
| 2 | Contract Info | Title, org, agreement type, dates, value, scope objective |
| 3 | Estimates | Service/material spend, total value, HR requirements |
| 4 | IKTVA | In-Kingdom Total Value Add targets, local content plan |
| 5 | Justifications | Technical, commercial, vendor, risk, compliance narrative |
| — | PDF | Generates branded A4 PDF SOW with all sections |

---

## Setup

```bash
cp .env.example .env   # add OPENAI_API_KEY and LLAMAPARSE_API_KEY
docker-compose up --build
# → http://localhost:8000
```

### Local (no Docker)
```bash
cd backend
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
export LLAMAPARSE_API_KEY=llx-...
uvicorn main:app --reload --port 8000
```

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/session` | Create / restore session |
| GET | `/api/session/{id}/data` | Get all SOW sections |
| POST | `/api/session/{id}/save` | Save a section |
| POST | `/api/session/{id}/parse-requirements` | Upload & parse requirement doc |
| GET | `/api/session/{id}/generate-pdf` | Stream PDF download |
| POST | `/api/chat` | Chat with SOW assistant |
| GET | `/api/chat/{id}/history` | Get chat history |

---

## PDF Structure

1. Cover page — contract title, org, dates, summary KPI table
2. Section 1 — Contract Information
3. Section 2 — Scope of Work & Deliverables
4. Section 3 — Commercial Estimates (spend table + HR)
5. Section 4 — IKTVA Targets
6. Section 5 — Justifications & Rationale
7. Section 6 — Approval & Signature Block

---

## Stack

| Layer | Tech |
|-------|------|
| Frontend | Vanilla HTML/CSS/JS |
| Backend | FastAPI (Python) |
| Doc parsing | LlamaParse |
| LLM | GPT-4o |
| Embeddings | all-MiniLM-L6-v2 |
| Vector DB | FAISS (CPU) |
| Relational DB | SQLite |
| PDF generation | ReportLab |
| Container | Docker + Compose |
