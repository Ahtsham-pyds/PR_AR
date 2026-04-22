from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))


def generate_sow(context_text: str):

    prompt = f"""
You are an expert in drafting Scope of Work (SOW) documents.

Using the structured information below, generate a professional SOW.

Context:
{context_text}

Include:
- Overview
- Scope
- Technology Stack
- Duration
- Deliverables

Keep it concise and professional.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {"role": "system", "content": "You generate SOW documents."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content