from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import sqlite3
from extraction import run_extraction_from_row
from reconciliation import reconcile_claims
from neo4j_injest import ingest_claims
from pydantic import BaseModel
from graph import app as app_api


app = FastAPI()

# --- DB Setup ---
conn = sqlite3.connect("sow.db", check_same_thread=False)
cursor = conn.cursor()

def get_db():
    return sqlite3.connect("sow.db")

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
    conn = get_db()
    cursor = conn.cursor()
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
    sow_id =1 

    row = {
        "id": sow_id,
        "contract_type": sow.contract_type,
        "date": sow.date,
        "currency": sow.currency,
        "technology": sow.technology,
        "duration": sow.duration,
        "vendor": sow.vendor,
        "requirements": sow.requirements
    }

    # 🔥 NO DB READ
    claims = run_extraction_from_row(row)
    #print(claims)

    # 2. Reconcile
    claims = reconcile_claims(claims)
    #print(claims)

    # 3. Push to Neo4j
    ingest_claims(claims)

    conn.commit()

    return {"message": "SOW stored successfully"}




class ChatRequest(BaseModel):
    user_input: str


# ---------------------------------
# CHAT ENDPOINT
# ---------------------------------
@app_api.post("/chat")
def chat_endpoint(payload: ChatRequest):

    try:
        # Run LangGraph agent
        result = app_api.invoke({
            "user_input": payload.user_input,
            "intent": "",
            "extracted_claims": [],
            "graph_result": {},
            "vector_result": [],
            "final_response": ""
        })

        return {
            "status": "success",
            "response": result["final_response"]
        }

    except Exception as e:
        return {
            "status": "error",
            "response": str(e)
        }


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)