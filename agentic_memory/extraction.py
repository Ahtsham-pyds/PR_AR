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



def extract_from_llm(text: str, sow_id: str) -> List[Dict]:
    """
    LLM-based extraction with strict JSON output
    """

    if not text.strip():
        return []

    prompt = f"""
You are an information extraction system.

Extract structured information from the SOW text.

Return ONLY valid JSON (no explanation).

Schema:
[
  {{
    "predicate": "USES_TECH | DELIVERABLE | TIMELINE | RESOURCE | COST | LOCATION",
    "object": "string"
  }}
]

Rules:
- Do NOT hallucinate
- Only extract explicitly mentioned info
- Keep object short and normalized
- If nothing found, return []

Text:
{text}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": "You extract structured data."},
                {"role": "user", "content": prompt}
            ]
        )

        content = response.choices[0].message.content.strip()

        # 🔒 Safety: Ensure valid JSON
        extracted = safe_json_loads(content)

        claims = []
        for item in extracted:
            if "predicate" in item and "object" in item:
                claims.append({
                    "subject": sow_id,
                    "predicate": normalize_predicate(item["predicate"]),
                    "object": normalize_text(item["object"]),
                    "confidence": 0.65,  # slightly higher than rules
                    "source": "llm"
                })

        return claims

    except Exception as e:
        print(f"LLM extraction failed: {e}")
        return []


# -----------------------
# MAIN PIPELINE
# -----------------------
def run_extraction():
    cursor.execute("SELECT * FROM sow")
    rows = cursor.fetchall()

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


if __name__ == "__main__":
    claims = run_extraction()
    for c in claims:
        print(c)