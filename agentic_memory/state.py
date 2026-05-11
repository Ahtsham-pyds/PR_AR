from typing_extensions import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    user_input: str
    tool: str
    extracted_claims: List[Dict]
    graph_result: Dict[str, Any]
    vector_result: List[str]
    final_response: str