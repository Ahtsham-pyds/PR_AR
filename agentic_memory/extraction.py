import sqlite3
from typing import List, Dict
import re
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()



LLAMAPARSE_API_KEY = os.getenv("LLAMAPARSE_API_KEY", "")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
DB_PATH            = "sow_builder.db"
FAISS_INDEX_PATH   = "faiss_index.bin"
FAISS_META_PATH    = "faiss_meta.pkl"
EMBED_DIM          = 384

client =  OpenAI(api_key=OPENAI_API_KEY)

# -----------------------
# DB CONNECTION
# -----------------------
conn = sqlite3.connect("sow.db")
cursor = conn.cursor()


# -----------------------
# NORMALIZATION HELPERS
# -----------------------
def normalize_text(text: str) -> str:
    return text.strip().lower()


# -----------------------
# STRUCTURED EXTRACTION
# -----------------------
def extract_from_structured(row) -> List[Dict]:
    claims = []

    sow_id = f"SOW_{row[0]}"

    mapping = {
        "contract_type": ("HAS_CONTRACT_TYPE", row[1]),
        "date": ("HAS_DATE", row[2]),
        "currency": ("USES_CURRENCY", row[3]),
        "technology": ("USES_TECH", row[4]),
        "duration": ("HAS_DURATION", row[5]),
        "vendor": ("PREFERRED_VENDOR", row[6]),
    }

    for field, (predicate, value) in mapping.items():
        if value:
            claims.append({
                "subject": sow_id,
                "predicate": predicate,
                "object": normalize_text(str(value)),
                "confidence": 1.0,
                "source": "form"
            })

    return claims


# -----------------------
# RULE-BASED TEXT EXTRACTION
# -----------------------
def extract_from_text_rules(text: str, sow_id: str) -> List[Dict]:
    claims = []

    # Example: extract tech keywords
    tech_keywords = ["python", "spark", "aws", "gcp", "azure"]

    for tech in tech_keywords:
        if re.search(rf"\b{tech}\b", text.lower()):
            claims.append({
                "subject": sow_id,
                "predicate": "USES_TECH",
                "object": tech,
                "confidence": 0.7,
                "source": "rule"
            })

    return claims


# -----------------------
# LLM EXTRACTION
# -----------------------

def safe_json_loads(content: str):
    try:
        return json.loads(content)
    except:
        # Try to fix common issues
        content = content.replace("```json", "").replace("```", "")
        return json.loads(content)
    
ALLOWED_PREDICATES = {
    "uses_tech": "USES_TECH",
    "technology": "USES_TECH",
    "deliverable": "DELIVERABLE",
    "timeline": "TIMELINE",
    "resource": "RESOURCE",
    "cost": "COST",
    "location": "LOCATION"
}

def normalize_predicate(p):
    return ALLOWED_PREDICATES.get(p.lower(), p.upper())



def extract_from_llm(text: str, sow_id: str):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "Extract structured data."},
            {"role": "user", "content": text}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "sow_extraction",
                "schema": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "predicate": {"type": "string"},
                                    "object": {"type": "string"}
                                },
                                "required": ["predicate", "object"]
                            }
                        }
                    },
                    "required": ["items"]
                }
            }
        }
    )

    data = json.loads(response.choices[0].message.content)

    claims = []
    for item in data["items"]:
        claims.append({
            "subject": sow_id,
            "predicate": normalize_predicate(item["predicate"]),
            "object": normalize_text(item["object"]),
            "confidence": 0.7,
            "source": "llm"
        })

    return claims


# -----------------------
# MAIN PIPELINE
# -----------------------
def run_extraction(sow_id=None):
    """Extract claims for a specific SOW ID or all if None"""
    conn = sqlite3.connect("sow.db")
    cursor = conn.cursor()
    if sow_id:
        #cursor.execute("SELECT * FROM sow WHERE id=?", (sow_id,))
        cursor.execute("SELECT * FROM sow")
    else:
        cursor.execute("SELECT * FROM sow")

    rows = cursor.fetchall()
    
    print(f"Retrieved rows: {len(rows)}") 

    all_claims = []

    for row in rows:
        sow_id = f"SOW_{row[0]}"
        requirements_text = row[7] or ""

        # 1. Structured
        structured_claims = extract_from_structured(row)

        # 2. Rule-based
        rule_claims = extract_from_text_rules(requirements_text, sow_id)

        # 3. LLM-based
        llm_claims = extract_from_llm(requirements_text, sow_id)

        all_claims.extend(structured_claims + rule_claims + llm_claims)


    return all_claims

def run_extraction_from_row(row):
    """Same as run_extraction but takes a single row dict instead of querying DB"""
    sow_id = f"SOW_{row['id']}"

    structured_claims = extract_from_structured([
        row['id'],
        row['contract_type'],
        row['date'],
        row['currency'],
        row['technology'],
        row['duration'],
        row['vendor'],
        row['requirements']
    ])

    rule_claims = extract_from_text_rules(row["requirements"], sow_id)
    llm_claims = extract_from_llm(row["requirements"], sow_id)

    return structured_claims + rule_claims + llm_claims


# if __name__ == "__main__":
#     claims = run_extraction()
#     for c in claims:
#         print(c)