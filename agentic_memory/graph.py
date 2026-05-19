from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import *
from intent_detection import tool_router

workflow = StateGraph(AgentState)

workflow.add_node("warm_load", warm_load_node)
workflow.add_node("router", tool_router)
workflow.add_node("add_to_vector", add_to_vector_node)
workflow.add_node("extract", extract_update)
workflow.add_node("reconcile", reconcile_node)
workflow.add_node("update_graph", update_graph_node)
workflow.add_node("query_graph", query_graph_node)
workflow.add_node("search_vector", search_vector_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("warm_load")

def route_after_detect(state):
    tool = state["tool"]

    if tool == "update_sow":
        return "extract"
    elif tool == "query_sow":
        return "query_graph"
    elif tool == "generate_sow":
        return "generate"
    elif tool == "search_memory":
        return "search_vector"


workflow.add_conditional_edges(
    "router",
    route_after_detect,
    {
        "extract": "extract",
        "query_graph": "query_graph",
        "generate": "generate",
        "search_vector": "search_vector"
    }
)

workflow.add_edge("warm_load", "router")
workflow.add_edge("extract", "reconcile")
workflow.add_edge("reconcile", "update_graph")
workflow.add_edge("update_graph", "add_to_vector")
workflow.add_edge("add_to_vector", END)

workflow.add_edge("query_graph", "add_to_vector")
workflow.add_edge("add_to_vector", END)

workflow.add_edge("search_vector", "add_to_vector")
workflow.add_edge("add_to_vector", END)

workflow.add_edge("generate", "add_to_vector")
workflow.add_edge("add_to_vector", END)

app = workflow.compile()

# result = app.invoke({
#     "user_input": "Hello, what is the time duration of the current SOW?"
# })

# print(result["final_response"])