from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import *
from intent_detection import tool_router

workflow = StateGraph(AgentState)

workflow.add_node("router", tool_router)
workflow.add_node("extract", extract_update)
workflow.add_node("reconcile", reconcile_node)
workflow.add_node("update_graph", update_graph_node)
workflow.add_node("query_graph", query_graph_node)
workflow.add_node("search_vector", search_vector_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("router")

def route_after_detect(state):
    tool = state["tool"]

    if tool == "update":
        return "extract"
    elif tool == "query":
        return "query_graph"
    elif tool == "generate":
        return "generate"
    else:
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

workflow.add_edge("extract", "reconcile")
workflow.add_edge("reconcile", "update_graph")
workflow.add_edge("update_graph", END)

workflow.add_edge("query_graph", END)
workflow.add_edge("search_vector", END)
workflow.add_edge("generate", END)

app = workflow.compile()

# result = app.invoke({
#     "user_input": "Change duration to 6 months"
# })

# print(result["final_response"])