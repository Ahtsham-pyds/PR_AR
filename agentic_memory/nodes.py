from openai import OpenAI
import os

from llm_response_generator import generate_sow
from retrieval import format_context_for_llm, get_sow_context
from neo4j_injest import ingest_claims
from reconciliation import reconcile_claims
from extraction import extract_from_llm
from vector_store import vector_search

client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))

def detect_intent(state):
    text = state["user_input"].lower()

    if any(x in text for x in ["change", "update", "add", "remove"]):
        intent = "update"

    elif any(x in text for x in ["draft", "generate", "create sow"]):
        intent = "generate"

    elif any(x in text for x in ["what", "show", "current"]):
        intent = "query"

    else:
        intent = "memory"

    state["intent"] = intent
    return state

def extract_update(state):
    text = state["user_input"]

    claims = extract_from_llm(text, "SOW_1")

    state["extracted_claims"] = claims
    return state

def reconcile_node(state):
    claims = reconcile_claims(state["extracted_claims"])
    state["extracted_claims"] = claims
    return state


def update_graph_node(state):
    ingest_claims(state["extracted_claims"])
    state["final_response"] = "SOW updated successfully."
    return state

def query_graph_node(state):
    result = get_sow_context("SOW_1")
    state["graph_result"] = result
    state["final_response"] = str(result)
    return state

def generate_node(state):
    context = get_sow_context("SOW_1")
    text = format_context_for_llm(context)

    doc = generate_sow(text)

    state["final_response"] = doc
    return state

def search_vector_node(state):
    docs = vector_search(state["user_input"])
    state["vector_result"] = docs
    state["final_response"] = "\n".join(docs)
    return state