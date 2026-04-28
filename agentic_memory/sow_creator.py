from retrieval import get_sow_context
from llm_response_generator import generate_sow
from retrieval import format_context_for_llm

sow_id = "SOW_1"

context = get_sow_context(sow_id)
#print(context)

context_text = format_context_for_llm(context)
#print( context_text)

document = generate_sow(context_text)

with open("generated_sow.txt", "w") as f:
    f.write(document)