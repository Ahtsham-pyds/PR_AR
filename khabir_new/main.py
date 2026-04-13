import os, json, uuid, sqlite3, pickle, io, tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
import faiss
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import httpx

# ── reportlab ─────────────────────────────────────────────────────────────────
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)

# ── Config ─────────────────────────────────────────────────────────────────────
LLAMAPARSE_API_KEY = os.getenv("LLAMAPARSE_API_KEY", "")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
DB_PATH            = "sow_builder.db"
FAISS_INDEX_PATH   = "faiss_index.bin"
FAISS_META_PATH    = "faiss_meta.pkl"
EMBED_DIM          = 384

openai_client = OpenAI(api_key=OPENAI_API_KEY)

embed_model: SentenceTransformer = None
faiss_index: faiss.IndexFlatL2   = None
faiss_meta:  List[Dict]          = []

# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global embed_model, faiss_index, faiss_meta
    print("Loading MiniLM …")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    if Path(FAISS_INDEX_PATH).exists():
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(FAISS_META_PATH, "rb") as f:
            faiss_meta = pickle.load(f)
    else:
        faiss_index = faiss.IndexFlatL2(EMBED_DIM)
        faiss_meta  = []
    _init_db()
    print("Ready.")
    yield
    _save_faiss()

app = FastAPI(title="SOW Builder API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── DB ─────────────────────────────────────────────────────────────────────────
def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _init_db():
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS sow_data (
                session_id          TEXT PRIMARY KEY,
                requirements_json   TEXT DEFAULT '{}',
                contract_json       TEXT DEFAULT '{}',
                estimates_json      TEXT DEFAULT '{}',
                iktva_json          TEXT DEFAULT '{}',
                justifications_json TEXT DEFAULT '{}',
                parsed_requirements TEXT DEFAULT '',
                updated_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            );
            CREATE TABLE IF NOT EXISTS chat_history (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role       TEXT,
                content    TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)

# ── FAISS ──────────────────────────────────────────────────────────────────────
def _save_faiss():
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    with open(FAISS_META_PATH, "wb") as f:
        pickle.dump(faiss_meta, f)

def _upsert_vectors(session_id: str, chunks: List[str], source: str):
    global faiss_meta, faiss_index
    faiss_meta_new = [m for m in faiss_meta if not (m["session_id"] == session_id and m["source"] == source)]
    faiss_index = faiss.IndexFlatL2(EMBED_DIM)
    if faiss_meta_new:
        faiss_index.add(np.array([m["vec"] for m in faiss_meta_new], dtype="float32"))
    faiss_meta = faiss_meta_new
    for chunk in chunks:
        if not chunk.strip():
            continue
        vec = embed_model.encode([chunk], normalize_embeddings=True)[0].astype("float32")
        faiss_index.add(vec.reshape(1, -1))
        faiss_meta.append({"session_id": session_id, "source": source, "text": chunk, "vec": vec})
    _save_faiss()

def _retrieve(session_id: str, query: str, k: int = 6) -> List[str]:
    if faiss_index.ntotal == 0:
        return []
    qvec = embed_model.encode([query], normalize_embeddings=True)[0].astype("float32").reshape(1, -1)
    _, indices = faiss_index.search(qvec, min(k * 3, max(faiss_index.ntotal, 1)))
    results = []
    for idx in indices[0]:
        if 0 <= idx < len(faiss_meta) and faiss_meta[idx]["session_id"] == session_id:
            results.append(faiss_meta[idx]["text"])
            if len(results) >= k:
                break
    return results

def _dict_to_chunks(section: str, data: Dict) -> List[str]:
    parts = " | ".join(f"{k}: {v}" for k, v in data.items() if v and str(v).strip())
    return [f"[{section.upper()}] {parts}"] if parts.strip() else []

# ── Schemas ────────────────────────────────────────────────────────────────────
class SessionCreate(BaseModel):
    session_id: Optional[str] = None

class SowSection(BaseModel):
    section: str
    data: Dict[str, Any]

class ChatMessage(BaseModel):
    session_id: str
    message: str

# ── Session ────────────────────────────────────────────────────────────────────
@app.post("/api/session")
def create_session(body: SessionCreate):
    sid = body.session_id or str(uuid.uuid4())
    with _get_conn() as conn:
        conn.execute("INSERT OR IGNORE INTO sessions(session_id) VALUES(?)", (sid,))
        conn.execute("INSERT OR IGNORE INTO sow_data(session_id) VALUES(?)", (sid,))
    return {"session_id": sid}

@app.get("/api/session/{session_id}/data")
def get_sow_data(session_id: str):
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM sow_data WHERE session_id=?", (session_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Session not found")
    return {
        "session_id":           session_id,
        "requirements":         json.loads(row["requirements_json"]),
        "contract":             json.loads(row["contract_json"]),
        "estimates":            json.loads(row["estimates_json"]),
        "iktva":                json.loads(row["iktva_json"]),
        "justifications":       json.loads(row["justifications_json"]),
        "parsed_requirements":  row["parsed_requirements"],
    }

# ── Save section ───────────────────────────────────────────────────────────────
COL_MAP = {
    "requirements":   "requirements_json",
    "contract":       "contract_json",
    "estimates":      "estimates_json",
    "iktva":          "iktva_json",
    "justifications": "justifications_json",
}

@app.post("/api/session/{session_id}/save")
def save_section(session_id: str, body: SowSection):
    col = COL_MAP.get(body.section)
    if not col:
        raise HTTPException(400, "Invalid section")
    with _get_conn() as conn:
        conn.execute("INSERT OR IGNORE INTO sessions(session_id) VALUES(?)", (session_id,))
        conn.execute("INSERT OR IGNORE INTO sow_data(session_id) VALUES(?)", (session_id,))
        conn.execute(f"UPDATE sow_data SET {col}=?, updated_at=CURRENT_TIMESTAMP WHERE session_id=?",
                     (json.dumps(body.data), session_id))
    _upsert_vectors(session_id, _dict_to_chunks(body.section, body.data), source=body.section)
    return {"status": "saved"}

# ── Parse requirement document ─────────────────────────────────────────────────
@app.post("/api/session/{session_id}/parse-requirements")
async def parse_requirements(session_id: str, file: UploadFile = File(...)):
    content = await file.read()
    raw_text = ""

    # Try LlamaParse first
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            upload_resp = await client.post(
                "https://api.cloud.llamaindex.ai/api/parsing/upload",
                headers={"Authorization": f"Bearer {LLAMAPARSE_API_KEY}"},
                files={"file": (file.filename, content, file.content_type)},
                data={"language": "en"}
            )
            upload_resp.raise_for_status()
            job_id = upload_resp.json()["id"]
            import asyncio
            for _ in range(30):
                await asyncio.sleep(3)
                s = await client.get(
                    f"https://api.cloud.llamaindex.ai/api/parsing/job/{job_id}",
                    headers={"Authorization": f"Bearer {LLAMAPARSE_API_KEY}"},
                )
                if s.json().get("status") == "SUCCESS":
                    break
            r = await client.get(
                f"https://api.cloud.llamaindex.ai/api/parsing/job/{job_id}/result/markdown",
                headers={"Authorization": f"Bearer {LLAMAPARSE_API_KEY}"},
            )
            raw_text = r.json().get("markdown", "")
    except Exception as e:
        print(f"LlamaParse fallback: {e}")
        raw_text = content.decode("utf-8", errors="ignore")

    # Extract SOW requirement hints via GPT-4o
    prompt = f"""You are an expert in Scope of Work documents for Saudi Arabia energy / construction contracts.

From the requirement document below, extract structured information and return ONLY valid JSON with this schema:

{{
  "contract": {{
    "contract_title": "",
    "organization": "",
    "agreement_type": "",
    "effective_date": "",
    "expiry_date": "",
    "scope_objective": "",
    "contract_value": ""
  }},
  "estimates": {{
    "service_spend": "",
    "material_spend": "",
    "total_spend": "",
    "currency": "SAR",
    "hr_required": "",
    "hr_roles": "",
    "duration_months": ""
  }},
  "iktva": {{
    "iktva_target_percentage": "",
    "iktva_category": "",
    "local_content_plan": "",
    "saudi_manpower_percentage": "",
    "in_kingdom_spend": ""
  }},
  "requirements_summary": "",
  "key_deliverables": "",
  "special_conditions": ""
}}

Fill every field you can find. Leave blank if not mentioned. Return ONLY the JSON.

DOCUMENT:
{raw_text[:7000]}
"""

    resp = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw_json = resp.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
    extracted = json.loads(raw_json)

    contract   = extracted.get("contract", {})
    estimates  = extracted.get("estimates", {})
    iktva      = extracted.get("iktva", {})
    req_summary = extracted.get("requirements_summary", "")
    deliverables = extracted.get("key_deliverables", "")
    special      = extracted.get("special_conditions", "")

    reqs_data = {
        "requirements_summary": req_summary,
        "key_deliverables":     deliverables,
        "special_conditions":   special,
        "source_filename":      file.filename,
    }

    with _get_conn() as conn:
        conn.execute("INSERT OR IGNORE INTO sessions(session_id) VALUES(?)", (session_id,))
        conn.execute("INSERT OR IGNORE INTO sow_data(session_id) VALUES(?)", (session_id,))
        conn.execute("""UPDATE sow_data SET
            requirements_json=?, contract_json=?, estimates_json=?, iktva_json=?,
            parsed_requirements=?, updated_at=CURRENT_TIMESTAMP
            WHERE session_id=?""",
            (json.dumps(reqs_data), json.dumps(contract), json.dumps(estimates),
             json.dumps(iktva), raw_text[:4000], session_id))

    for sec, d in [("requirements", reqs_data), ("contract", contract),
                   ("estimates", estimates), ("iktva", iktva)]:
        _upsert_vectors(session_id, _dict_to_chunks(sec, d), source=sec)
    _upsert_vectors(session_id, [f"[FULL DOCUMENT] {raw_text[:3000]}"], source="document")

    return {
        "session_id": session_id,
        "requirements": reqs_data,
        "contract":     contract,
        "estimates":    estimates,
        "iktva":        iktva,
    }

# ── Chat ───────────────────────────────────────────────────────────────────────
@app.post("/api/chat")
def chat(body: ChatMessage):
    context_chunks = _retrieve(body.session_id, body.message, k=6)
    context = "\n".join(context_chunks) or "No SOW data saved yet."

    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content FROM chat_history WHERE session_id=? ORDER BY id DESC LIMIT 10",
            (body.session_id,)
        ).fetchall()
    history = [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    system = f"""You are an expert SOW (Scope of Work) assistant for Saudi Arabia energy and construction contracts.
You help users build compliant, professional SOW documents. You understand IKTVA (In-Kingdom Total Value Add) requirements, Saudi Aramco contracting standards, and general project scoping best practices.

Retrieved SOW context for this session:
{context}

Answer questions clearly and professionally. If asked about specific figures, clauses, or content, refer to the data above. Suggest improvements when appropriate."""

    messages = [{"role": "system", "content": system}] + history + [{"role": "user", "content": body.message}]
    resp = openai_client.chat.completions.create(model="gpt-4o", messages=messages, temperature=0.7, max_tokens=600)
    reply = resp.choices[0].message.content

    with _get_conn() as conn:
        conn.execute("INSERT OR IGNORE INTO sessions(session_id) VALUES(?)", (body.session_id,))
        conn.execute("INSERT INTO chat_history(session_id,role,content) VALUES(?,?,?)", (body.session_id, "user", body.message))
        conn.execute("INSERT INTO chat_history(session_id,role,content) VALUES(?,?,?)", (body.session_id, "assistant", reply))

    return {"reply": reply}

@app.get("/api/chat/{session_id}/history")
def get_history(session_id: str):
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content, created_at FROM chat_history WHERE session_id=? ORDER BY id",
            (session_id,)
        ).fetchall()
    return [{"role": r["role"], "content": r["content"], "time": r["created_at"]} for r in rows]

# ── Generate PDF SOW ───────────────────────────────────────────────────────────
@app.get("/api/session/{session_id}/generate-pdf")
def generate_pdf(session_id: str):
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM sow_data WHERE session_id=?", (session_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Session not found")

    contract       = json.loads(row["contract_json"])
    estimates      = json.loads(row["estimates_json"])
    iktva          = json.loads(row["iktva_json"])
    justifications = json.loads(row["justifications_json"])
    requirements   = json.loads(row["requirements_json"])

    buf = io.BytesIO()
    _build_pdf(buf, contract, estimates, iktva, justifications, requirements, session_id)
    buf.seek(0)

    filename = f"SOW_{contract.get('contract_title','Document').replace(' ','_')[:40]}.pdf"
    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )

# ── PDF Builder ────────────────────────────────────────────────────────────────
def _v(d: Dict, k: str, default: str = "—") -> str:
    return str(d.get(k, "") or "").strip() or default

def _build_pdf(buf, contract, estimates, iktva, justifications, requirements, session_id):
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=22*mm, bottomMargin=22*mm,
    )

    # ── Colour palette ─────────────────────────────────────────────────────────
    DARK_TEAL   = colors.HexColor("#0a4d5c")
    MID_TEAL    = colors.HexColor("#0e7a8a")
    LIGHT_TEAL  = colors.HexColor("#e0f4f7")
    GOLD        = colors.HexColor("#c9a84c")
    LIGHT_GOLD  = colors.HexColor("#fdf6e3")
    DARK_GRAY   = colors.HexColor("#2d2d2d")
    MID_GRAY    = colors.HexColor("#666666")
    LIGHT_GRAY  = colors.HexColor("#f5f5f5")
    WHITE       = colors.white

    # ── Styles ─────────────────────────────────────────────────────────────────
    base = getSampleStyleSheet()

    def S(name, **kw):
        s = ParagraphStyle(name, **kw)
        return s

    cover_title = S("ct", fontName="Helvetica-Bold", fontSize=26, textColor=WHITE,
                    alignment=TA_CENTER, spaceAfter=6, leading=32)
    cover_sub   = S("cs", fontName="Helvetica", fontSize=13, textColor=colors.HexColor("#c8e8ed"),
                    alignment=TA_CENTER, spaceAfter=4)
    cover_meta  = S("cm", fontName="Helvetica", fontSize=10, textColor=GOLD,
                    alignment=TA_CENTER, spaceAfter=4)

    sec_heading = S("sh", fontName="Helvetica-Bold", fontSize=13, textColor=WHITE,
                    backColor=DARK_TEAL, alignment=TA_LEFT, spaceAfter=0,
                    spaceBefore=14, leading=18,
                    leftIndent=-2, rightIndent=-2)
    sub_heading = S("subh", fontName="Helvetica-Bold", fontSize=11, textColor=DARK_TEAL,
                    spaceBefore=10, spaceAfter=4)
    body        = S("body", fontName="Helvetica", fontSize=9.5, textColor=DARK_GRAY,
                    leading=14, spaceAfter=6, alignment=TA_JUSTIFY)
    label       = S("lbl", fontName="Helvetica-Bold", fontSize=9, textColor=MID_TEAL,
                    spaceAfter=1)
    value_style = S("val", fontName="Helvetica", fontSize=9.5, textColor=DARK_GRAY,
                    leading=13, spaceAfter=8)
    footer_style= S("ft", fontName="Helvetica", fontSize=8, textColor=MID_GRAY,
                    alignment=TA_CENTER)
    tbl_hdr     = S("th2", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE)
    tbl_cell    = S("tc", fontName="Helvetica", fontSize=9, textColor=DARK_GRAY, leading=12)

    story = []

    # ═══════════════════════════════════════════════════════════
    # COVER PAGE
    # ═══════════════════════════════════════════════════════════
    # Dark teal header block via table trick
    cover_data = [
        [Paragraph("SCOPE OF WORK", cover_title)],
        [Paragraph(_v(contract, "contract_title", "Contract Title"), cover_sub)],
        [Spacer(1, 6)],
        [Paragraph(_v(contract, "organization", "Organisation"), cover_meta)],
        [Paragraph(f"Agreement Type: {_v(contract, 'agreement_type')}", cover_meta)],
        [Paragraph(f"Effective Date: {_v(contract, 'effective_date')}  |  Expiry: {_v(contract, 'expiry_date')}", cover_meta)],
    ]
    cover_tbl = Table(cover_data, colWidths=[170*mm])
    cover_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), DARK_TEAL),
        ("TOPPADDING",   (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
        ("LEFTPADDING",  (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
    ]))
    story.append(cover_tbl)
    story.append(Spacer(1, 10*mm))

    # Gold accent bar
    story.append(HRFlowable(width="100%", thickness=3, color=GOLD, spaceAfter=8))

    # Summary card
    summary_rows = [
        ["Contract Value", _v(estimates, "total_spend") + " " + _v(estimates, "currency", "SAR"),
         "Duration", _v(estimates, "duration_months") + " months"],
        ["IKTVA Target", _v(iktva, "iktva_target_percentage") + "%",
         "Category", _v(iktva, "iktva_category")],
        ["In-Kingdom Spend", _v(iktva, "in_kingdom_spend"),
         "HR Required", _v(estimates, "hr_required")],
    ]
    st = Table(summary_rows, colWidths=[38*mm, 47*mm, 38*mm, 47*mm])
    st.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (0, -1), LIGHT_TEAL),
        ("BACKGROUND",   (2, 0), (2, -1), LIGHT_TEAL),
        ("BACKGROUND",   (1, 0), (1, -1), WHITE),
        ("BACKGROUND",   (3, 0), (3, -1), WHITE),
        ("FONTNAME",     (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",     (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR",    (0, 0), (0, -1), DARK_TEAL),
        ("TEXTCOLOR",    (2, 0), (2, -1), DARK_TEAL),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#c0d8dc")),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("LEFTPADDING",  (0, 0), (-1, -1), 7),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(st)
    story.append(Spacer(1, 6*mm))

    gen_date = datetime.now().strftime("%d %B %Y")
    story.append(Paragraph(f"Document generated: {gen_date}  |  Session: {session_id[:8]}…", footer_style))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # HELPER: section header
    # ═══════════════════════════════════════════════════════════
    def section(title):
        story.append(Spacer(1, 2*mm))
        hdr = Table([[Paragraph(f"  {title}", sec_heading)]], colWidths=[170*mm])
        hdr.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, -1), DARK_TEAL),
            ("TOPPADDING",   (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 7),
            ("LEFTPADDING",  (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ]))
        story.append(hdr)
        story.append(Spacer(1, 2*mm))

    def kv(lbl_text, val_text):
        story.append(KeepTogether([
            Paragraph(lbl_text, label),
            Paragraph(val_text or "—", value_style),
        ]))

    def two_col(pairs):
        rows = []
        it = iter(pairs)
        for a in it:
            b = next(it, ("", ""))
            rows.append([
                Paragraph(a[0], label), Paragraph(a[1] or "—", value_style),
                Paragraph(b[0], label), Paragraph(b[1] or "—", value_style),
            ])
        if not rows:
            return
        t = Table(rows, colWidths=[32*mm, 53*mm, 32*mm, 53*mm])
        t.setStyle(TableStyle([
            ("VALIGN",       (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING",   (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
            ("LEFTPADDING",  (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("LINEBELOW",    (0, 0), (-1, -2), 0.3, colors.HexColor("#e0e0e0")),
        ]))
        story.append(t)

    # ═══════════════════════════════════════════════════════════
    # 1 · CONTRACT INFORMATION
    # ═══════════════════════════════════════════════════════════
    section("1.  Contract Information")
    two_col([
        ("Contract Title",     _v(contract, "contract_title")),
        ("Organisation",       _v(contract, "organization")),
        ("Agreement Type",     _v(contract, "agreement_type")),
        ("Contract Value",     _v(contract, "contract_value")),
        ("Effective Date",     _v(contract, "effective_date")),
        ("Expiry Date",        _v(contract, "expiry_date")),
    ])
    kv("Scope Objective", _v(contract, "scope_objective"))

    # ═══════════════════════════════════════════════════════════
    # 2 · SCOPE OF WORK & DELIVERABLES
    # ═══════════════════════════════════════════════════════════
    section("2.  Scope of Work &amp; Deliverables")
    kv("Requirements Summary", _v(requirements, "requirements_summary"))
    kv("Key Deliverables",     _v(requirements, "key_deliverables"))
    kv("Special Conditions",   _v(requirements, "special_conditions"))

    # ═══════════════════════════════════════════════════════════
    # 3 · COMMERCIAL ESTIMATES
    # ═══════════════════════════════════════════════════════════
    section("3.  Commercial Estimates")
    currency = _v(estimates, "currency", "SAR")
    spend_data = [
        [Paragraph("Cost Component", tbl_hdr), Paragraph("Amount", tbl_hdr), Paragraph("Notes", tbl_hdr)],
        [Paragraph("Service Spend", tbl_cell),
         Paragraph(f"{currency} {_v(estimates, 'service_spend')}", tbl_cell),
         Paragraph("Labour, consulting & services", tbl_cell)],
        [Paragraph("Material Spend", tbl_cell),
         Paragraph(f"{currency} {_v(estimates, 'material_spend')}", tbl_cell),
         Paragraph("Equipment, materials & supply", tbl_cell)],
        [Paragraph("Total Contract Value", tbl_hdr),
         Paragraph(f"{currency} {_v(estimates, 'total_spend')}", tbl_hdr),
         Paragraph("", tbl_cell)],
    ]
    spend_tbl = Table(spend_data, colWidths=[55*mm, 55*mm, 60*mm])
    spend_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), DARK_TEAL),
        ("BACKGROUND",   (0, 3), (-1, 3), LIGHT_TEAL),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#b0cfd4")),
        ("ROWBACKGROUNDS",(0, 1), (-1, 2), [WHITE, LIGHT_GRAY]),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("LEFTPADDING",  (0, 0), (-1, -1), 7),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(spend_tbl)
    story.append(Spacer(1, 4*mm))

    # HR table
    story.append(Paragraph("Human Resources", sub_heading))
    two_col([
        ("No. of Personnel",  _v(estimates, "hr_required")),
        ("Contract Duration", _v(estimates, "duration_months") + " months"),
        ("Key Roles",         _v(estimates, "hr_roles")),
        ("",                  ""),
    ])

    # ═══════════════════════════════════════════════════════════
    # 4 · IKTVA
    # ═══════════════════════════════════════════════════════════
    section("4.  IKTVA (In-Kingdom Total Value Add)")
    iktva_data = [
        [Paragraph("Parameter", tbl_hdr), Paragraph("Target / Value", tbl_hdr)],
        [Paragraph("IKTVA Target (%)", tbl_cell),    Paragraph(_v(iktva, "iktva_target_percentage") + "%", tbl_cell)],
        [Paragraph("IKTVA Category", tbl_cell),       Paragraph(_v(iktva, "iktva_category"), tbl_cell)],
        [Paragraph("In-Kingdom Spend", tbl_cell),     Paragraph(f"{currency} {_v(iktva, 'in_kingdom_spend')}", tbl_cell)],
        [Paragraph("Saudi Manpower (%)", tbl_cell),   Paragraph(_v(iktva, "saudi_manpower_percentage") + "%", tbl_cell)],
    ]
    iktva_tbl = Table(iktva_data, colWidths=[90*mm, 80*mm])
    iktva_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), DARK_TEAL),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#b0cfd4")),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, LIGHT_TEAL]),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(iktva_tbl)
    story.append(Spacer(1, 4*mm))
    kv("Local Content Plan", _v(iktva, "local_content_plan"))

    # ═══════════════════════════════════════════════════════════
    # 5 · JUSTIFICATIONS
    # ═══════════════════════════════════════════════════════════
    section("5.  Justifications &amp; Supporting Rationale")

    just_fields = [
        ("technical_justification",   "Technical Justification"),
        ("commercial_justification",  "Commercial Justification"),
        ("vendor_selection",          "Vendor / Supplier Selection Rationale"),
        ("risk_assessment",           "Risk Assessment"),
        ("compliance_notes",          "Regulatory &amp; Compliance Notes"),
        ("additional_notes",          "Additional Notes"),
    ]
    for key, heading in just_fields:
        val = _v(justifications, key)
        if val and val != "—":
            story.append(Paragraph(heading, sub_heading))
            story.append(Paragraph(val, body))

    # ═══════════════════════════════════════════════════════════
    # 6 · SIGNATURE BLOCK
    # ═══════════════════════════════════════════════════════════
    story.append(PageBreak())
    section("6.  Approval &amp; Signatures")
    story.append(Spacer(1, 6*mm))

    sig_data = [
        [Paragraph("Role", tbl_hdr), Paragraph("Name", tbl_hdr),
         Paragraph("Signature", tbl_hdr), Paragraph("Date", tbl_hdr)],
        ["Prepared by", "", "", ""],
        ["Reviewed by", "", "", ""],
        ["Approved by", "", "", ""],
        ["Contractor Rep", "", "", ""],
    ]
    sig_tbl = Table(sig_data, colWidths=[38*mm, 48*mm, 48*mm, 36*mm])
    sig_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), DARK_TEAL),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#b0cfd4")),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
        ("TOPPADDING",   (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 14),
        ("LEFTPADDING",  (0, 0), (-1, -1), 7),
        ("FONTNAME",     (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 1), (0, -1), 8.5),
        ("TEXTCOLOR",    (0, 1), (0, -1), DARK_TEAL),
    ]))
    story.append(sig_tbl)
    story.append(Spacer(1, 10*mm))
    story.append(HRFlowable(width="100%", thickness=1, color=GOLD))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        f"This Scope of Work was generated by SOW Forge  |  {gen_date}  |  Confidential",
        footer_style
    ))

    doc.build(story)

# ── Serve frontend ─────────────────────────────────────────────────────────────
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="static")
