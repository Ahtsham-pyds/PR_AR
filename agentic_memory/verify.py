from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    query:str
    data:str
    status:str
    
def router_node(state:AgentState):
    if state["query"] in ["get","show"]:
        return "tool1_result"
    elif state["query"] in ["update","modify"]:
        return "tool2_result"
    
        
def tool1_node(state:AgentState):
    return "tool1 result"

def tool2_node(state: AgentState):
    return "tool2 result"

workflow =  StateGraph(AgentState)


workflow.add_node("router", router_node)
workflow.add_node("tool1",tool1_node)
workflow.add_node("tool2",tool2_node)

workflow.set_entry_point("router")   

workflow.add_conditional_edges("router",router_node,{"tool1":"tool1_result","tool2":"tool2_result"})
workflow.add_edge("tool1",END)
workflow.add_edge("tool2",END)

app = workflow.compile()


app.invoke({
    "query":"get"})