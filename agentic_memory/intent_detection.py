from openai import OpenAI
import os
import json

client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))
from dotenv import load_dotenv
load_dotenv()


TOOLS = [
    {
        "name": "update_sow",
        "description": "Update structured SOW fields like duration, vendor, technology",
    },
    {
        "name": "query_sow",
        "description": "Get current structured SOW information like duration, vendor",
    },
    {
        "name": "search_memory",
        "description": "Search past conversations and notes",
    },
    {
        "name": "generate_sow",
        "description": "Generate a full SOW document"
    }
]




def tool_router(state):

    user_input = state["user_input"]

    prompt = f"""
You are a routing agent.

Decide which tool to use based on user input.

Available tools:
{json.dumps(TOOLS, indent=2)}

Return ONLY JSON:

{{
  "tool": "tool_name",
  "confidence": 0-1,
  "reason": "short explanation"
}}

Rules:
- Pick ONLY one tool
- High confidence only if clear intent
- If ambiguous, pick best guess but lower confidence
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "You route user requests to tools."},
            {"role": "user", "content": prompt + "\n\nUser Input:\n" + user_input}
        ]
    )

    content = response.choices[0].message.content.strip()

    try:
        decision = json.loads(content)
    except:
        decision = {
            "tool": "search_memory",
            "confidence": 0.3,
            "reason": "fallback"
        }

    state["tool"] = decision["tool"]
    state["confidence"] = decision["confidence"]
    
    print('printing state in router',state)

    return state

