import os
import json
import uuid
import sqlite3
import pickle
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import numpy as np
import faiss
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import httpx

# ── Config ────────────────────────────────────────────────────────────────────
LLAMAPARSE_API_KEY = os.getenv("LLAMAPARSE_API_KEY", "")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
DB_PATH            = "profile_builder.db"
FAISS_INDEX_PATH   = "faiss_index.bin"
FAISS_META_PATH    = "faiss_meta.pkl"
EMBED_DIM          = 384   # all-MiniLM-L6-v2

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ── Globals (loaded at startup) ───────────────────────────────────────────────
embed_model: SentenceTransformer = None
faiss_index: faiss.IndexFlatL2   = None
faiss_meta:  List[Dict]          = []   # [{session_id, text, source}, ...]

# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global embed_model, faiss_index, faiss_meta
    print("Loading MiniLM embedding model …")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    if Path(FAISS_INDEX_PATH).exists():
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(FAISS_META_PATH, "rb") as f:
            faiss_meta = pickle.load(f)
    else:
        faiss_index = faiss.IndexFlatL2(EMBED_DIM)
        faiss_meta  = []

    _init_db()
    print("Startup complete.")
    yield
    # persist on shutdown
    _save_faiss()

app = FastAPI(title="Profile Builder API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── SQLite helpers ─────────────────────────────────────────────────────────────
def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _init_db():
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id   TEXT PRIMARY KEY,
                created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS profiles (
                session_id      TEXT PRIMARY KEY,
                personal_json   TEXT DEFAULT '{}',
                education_json  TEXT DEFAULT '{}',
                experience_json TEXT DEFAULT '{}',
                updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            );
            CREATE TABLE IF NOT EXISTS chat_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT,
                role        TEXT,
                content     TEXT,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            );
        """)

# ── FAISS helpers ──────────────────────────────────────────────────────────────
def _save_faiss():
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    with open(FAISS_META_PATH, "wb") as f:
        pickle.dump(faiss_meta, f)

def _upsert_vectors(session_id: str, chunks: List[str], source: str):
    global faiss_meta
    # remove old chunks for this session + source
    faiss_meta_new = [m for m in faiss_meta if not (m["session_id"] == session_id and m["source"] == source)]
    # rebuild index from scratch (simple approach for small scale)
    global faiss_index
    faiss_index = faiss.IndexFlatL2(EMBED_DIM)
    if faiss_meta_new:
        old_vecs = np.array([m["vec"] for m in faiss_meta_new], dtype="float32")
        faiss_index.add(old_vecs)
    faiss_meta = faiss_meta_new

    for chunk in chunks:
        vec = embed_model.encode([chunk], normalize_embeddings=True)[0].astype("float32")
        faiss_index.add(vec.reshape(1, -1))
        faiss_meta.append({"session_id": session_id, "source": source, "text": chunk, "vec": vec})
    _save_faiss()

def _retrieve(session_id: str, query: str, k: int = 5) -> List[str]:
    if faiss_index.ntotal == 0:
        return []
    qvec = embed_model.encode([query], normalize_embeddings=True)[0].astype("float32").reshape(1, -1)
    distances, indices = faiss_index.search(qvec, min(k * 3, faiss_index.ntotal))
    results = []
    for idx in indices[0]:
        if idx < 0 or idx >= len(faiss_meta):
            continue
        meta = faiss_meta[idx]
        if meta["session_id"] == session_id:
            results.append(meta["text"])
        if len(results) >= k:
            break
    return results

# ── Pydantic schemas ───────────────────────────────────────────────────────────
class SessionCreate(BaseModel):
    session_id: Optional[str] = None

class ProfileSection(BaseModel):
    section: str   # "personal" | "education" | "experience"
    data: Dict[str, Any]

class ChatMessage(BaseModel):
    session_id: str
    message: str

class ParseResponse(BaseModel):
    session_id: str
    personal:   Dict
    education:  Dict
    experience: Dict
    raw_text:   str

# ── Session endpoints ──────────────────────────────────────────────────────────
@app.post("/api/session")
def create_session(body: SessionCreate):
    sid = body.session_id or str(uuid.uuid4())
    with _get_conn() as conn:
        conn.execute("INSERT OR IGNORE INTO sessions(session_id) VALUES(?)", (sid,))
        conn.execute("INSERT OR IGNORE INTO profiles(session_id) VALUES(?)", (sid,))
    return {"session_id": sid}

@app.get("/api/session/{session_id}/profile")
def get_profile(session_id: str):
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM profiles WHERE session_id=?", (session_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Session not found")
    return {
        "session_id": session_id,
        "personal":   json.loads(row["personal_json"]),
        "education":  json.loads(row["education_json"]),
        "experience": json.loads(row["experience_json"]),
    }

# ── Profile save endpoint ──────────────────────────────────────────────────────
@app.post("/api/session/{session_id}/save")
def save_section(session_id: str, body: ProfileSection):
    col_map = {"personal": "personal_json", "education": "education_json", "experience": "experience_json"}
    col = col_map.get(body.section)
    if not col:
        raise HTTPException(400, "Invalid section")

    with _get_conn() as conn:
        conn.execute(f"INSERT OR IGNORE INTO profiles(session_id) VALUES(?)", (session_id,))
        conn.execute(f"UPDATE profiles SET {col}=?, updated_at=CURRENT_TIMESTAMP WHERE session_id=?",
                     (json.dumps(body.data), session_id))

    # update FAISS with new section text
    text_chunks = _dict_to_chunks(body.section, body.data)
    _upsert_vectors(session_id, text_chunks, source=body.section)
    return {"status": "saved"}

def _dict_to_chunks(section: str, data: Dict) -> List[str]:
    chunks = []
    flat = f"[{section.upper()}] " + " | ".join(f"{k}: {v}" for k, v in data.items() if v)
    if flat.strip():
        chunks.append(flat)
    return chunks

# ── LlamaParse document upload ─────────────────────────────────────────────────
@app.post("/api/session/{session_id}/parse-document")
async def parse_document(session_id: str, file: UploadFile = File(...)):
    content = await file.read()

    # ── Call LlamaParse ───────────────────────────────────────────────────────
    raw_text = ""
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

            # Poll for result
            import asyncio
            for _ in range(30):
                await asyncio.sleep(3)
                status_resp = await client.get(
                    f"https://api.cloud.llamaindex.ai/api/parsing/job/{job_id}",
                    headers={"Authorization": f"Bearer {LLAMAPARSE_API_KEY}"},
                )
                status = status_resp.json().get("status")
                if status == "SUCCESS":
                    break
                if status == "ERROR":
                    raise Exception("LlamaParse job failed")

            result_resp = await client.get(
                f"https://api.cloud.llamaindex.ai/api/parsing/job/{job_id}/result/markdown",
                headers={"Authorization": f"Bearer {LLAMAPARSE_API_KEY}"},
            )
            result_resp.raise_for_status()
            raw_text = result_resp.json().get("markdown", "")
    except Exception as e:
        print(f"LlamaParse error: {e}. Falling back to raw text.")
        try:
            raw_text = content.decode("utf-8", errors="ignore")
        except:
            raw_text = ""

    # ── Use GPT-4o to extract structured profile data ─────────────────────────
    extraction_prompt = f"""
You are a resume/CV parser. From the text below, extract structured information and return ONLY a valid JSON object with this exact schema:

{{
  "personal": {{
    "full_name": "",
    "email": "",
    "phone": "",
    "location": "",
    "linkedin": "",
    "website": "",
    "summary": ""
  }},
  "education": {{
    "degree": "",
    "institution": "",
    "field_of_study": "",
    "start_year": "",
    "end_year": "",
    "gpa": "",
    "achievements": ""
  }},
  "experience": {{
    "current_title": "",
    "current_company": "",
    "years_of_experience": "",
    "skills": "",
    "previous_roles": "",
    "certifications": ""
  }}
}}

Fill in all fields you can find. Leave blank if not found. Return ONLY the JSON, no commentary.

DOCUMENT TEXT:
{raw_text[:6000]}
"""

    gpt_resp = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": extraction_prompt}],
        temperature=0,
    )
    raw_json = gpt_resp.choices[0].message.content.strip()
    # strip markdown fences if any
    raw_json = raw_json.replace("```json", "").replace("```", "").strip()
    extracted = json.loads(raw_json)

    personal   = extracted.get("personal", {})
    education  = extracted.get("education", {})
    experience = extracted.get("experience", {})

    # ── Persist to DB ─────────────────────────────────────────────────────────
    with _get_conn() as conn:
        conn.execute("INSERT OR IGNORE INTO sessions(session_id) VALUES(?)", (session_id,))
        conn.execute("INSERT OR IGNORE INTO profiles(session_id) VALUES(?)", (session_id,))
        conn.execute("""UPDATE profiles SET
            personal_json=?, education_json=?, experience_json=?,
            updated_at=CURRENT_TIMESTAMP WHERE session_id=?""",
            (json.dumps(personal), json.dumps(education), json.dumps(experience), session_id))

    # ── Index all sections in FAISS ───────────────────────────────────────────
    chunks = (
        _dict_to_chunks("personal", personal) +
        _dict_to_chunks("education", education) +
        _dict_to_chunks("experience", experience) +
        [f"[FULL DOCUMENT] {raw_text[:3000]}"]
    )
    _upsert_vectors(session_id, chunks, source="document")

    return ParseResponse(
        session_id=session_id,
        personal=personal,
        education=education,
        experience=experience,
        raw_text=raw_text[:500],
    )

# ── Chat endpoint ──────────────────────────────────────────────────────────────
@app.post("/api/chat")
def chat(body: ChatMessage):
    session_id = body.session_id
    user_msg   = body.message

    # Retrieve relevant context from FAISS
    context_chunks = _retrieve(session_id, user_msg, k=5)
    context = "\n".join(context_chunks) if context_chunks else "No profile data available yet."

    # Load last 10 messages from DB for conversation memory
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content FROM chat_history WHERE session_id=? ORDER BY id DESC LIMIT 10",
            (session_id,)
        ).fetchall()
    history = [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    system_prompt = f"""You are a helpful profile assistant embedded in a profile builder application.
The user is building their professional profile across three sections: Personal Details, Education, and Experience.

Here is the relevant profile context retrieved for this query:
{context}

Answer questions based on this profile data. If asked to update or fill information, confirm what you'll update.
Be concise, friendly, and professional. If you don't have the information, say so clearly."""

    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": user_msg}]

    gpt_resp = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
        max_tokens=500,
    )
    assistant_reply = gpt_resp.choices[0].message.content

    # Persist to chat history
    with _get_conn() as conn:
        conn.execute("INSERT OR IGNORE INTO sessions(session_id) VALUES(?)", (session_id,))
        conn.execute("INSERT INTO chat_history(session_id, role, content) VALUES(?,?,?)",
                     (session_id, "user", user_msg))
        conn.execute("INSERT INTO chat_history(session_id, role, content) VALUES(?,?,?)",
                     (session_id, "assistant", assistant_reply))

    return {"reply": assistant_reply, "context_used": len(context_chunks) > 0}

@app.get("/api/chat/{session_id}/history")
def get_chat_history(session_id: str):
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content, created_at FROM chat_history WHERE session_id=? ORDER BY id",
            (session_id,)
        ).fetchall()
    return [{"role": r["role"], "content": r["content"], "time": r["created_at"]} for r in rows]

# ── Serve frontend ─────────────────────────────────────────────────────────────
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
