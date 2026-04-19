from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import sqlite3
from extraction import run_extraction
from reconciliation import reconcile_claims
from neo4j_injest import ingest_claims

app = FastAPI()

# --- DB Setup ---
conn = sqlite3.connect("sow.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS sow (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    contract_type TEXT,
    date TEXT,
    currency TEXT,
    technology TEXT,
    duration INTEGER,
    vendor TEXT,
    requirements TEXT
)
""")
conn.commit()


# --- Pydantic मॉडल ---
class SOW(BaseModel):
    contract_type: str
    date: str
    currency: str
    technology: str
    duration: int
    vendor: str
    requirements: str


# --- Serve Frontend ---
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    with open("frontend/index.html", "r") as f:
        return f.read()


# --- API to Save Data ---
@app.post("/submit")
def submit_sow(sow: SOW):
    cursor.execute("""
    INSERT INTO sow (
        contract_type, date, currency,
        technology, duration, vendor, requirements
    ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        sow.contract_type,
        sow.date,
        sow.currency,
        sow.technology,
        sow.duration,
        sow.vendor,
        sow.requirements
    ))
    
    sow_id = cursor.lastrowid
    claims = run_extraction(sow_id=sow_id)

    # 2. Reconcile
    claims = reconcile_claims(claims)

    # 3. Push to Neo4j
    ingest_claims(claims)

    conn.commit()

    return {"message": "SOW stored successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)