from neo4j import GraphDatabase

URI = "neo4j://127.0.0.1:7687"
USER = "neo4j"
PASSWORD = "password"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))


def get_sow_context(sow_id: str):
    query = """
    MATCH (s:SOW {id: $sow_id})-[r]->(e)
    WHERE r.is_active = true
    RETURN type(r) AS predicate, e.name AS value
    """

    context = {}

    with driver.session(database="sow1") as session:
        result = session.run(query, sow_id=sow_id)

        for record in result:
            pred = record["predicate"]
            val = record["value"]

            if pred not in context:
                context[pred] = []

            context[pred].append(val)

    return context


def format_context_for_llm(context: dict) -> str:
    lines = []

    for k, v in context.items():
        lines.append(f"{k}: {', '.join(v)}")

    return "\n".join(lines)