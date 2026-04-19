from neo4j import GraphDatabase
from reconciliation import reconcile_claims

URI = "neo4j://127.0.0.1:7687"
USER = "neo4j"
PASSWORD = "password"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))


# -----------------------
# HELPER: CREATE ENTITY
# -----------------------
def create_entity(tx, name):
    tx.run("""
        MERGE (e:Entity {name: $name})
    """, name=name)


# -----------------------
# HELPER: CREATE SOW NODE
# -----------------------
def create_sow(tx, sow_id):
    tx.run("""
        MERGE (s:SOW {id: $sow_id})
    """, sow_id=sow_id)


# -----------------------
# CREATE RELATIONSHIP
# -----------------------
def create_relationship(tx, sow_id, predicate, obj, confidence,sources):
    query = f"""
    MATCH (s:SOW {{id: $sow_id}})
    MATCH (e:Entity {{name: $obj}})
    MERGE (s)-[r:{predicate}]->(e)
    SET r.confidence = $confidence,
        r.sources = $sources
    """
    tx.run(
    query,
    sow_id=sow_id,
    obj=obj,
    confidence=confidence,
    sources=sources
)


# -----------------------
# MAIN INGEST FUNCTION
# -----------------------
def ingest_claims(claims):
    with driver.session() as session:
        for c in claims:
            sow_id = c["subject"]
            predicate = c["predicate"]
            obj = c["object"]
            confidence = c["confidence"]
            sources = c["sources"]

            session.execute_write(create_sow, sow_id)
            session.execute_write(create_entity, obj)
            session.execute_write(create_relationship, sow_id, predicate, obj, confidence,sources)


# -----------------------
# RUN
# -----------------------
if __name__ == "__main__":
    from extraction import run_extraction

    claims = run_extraction()
    claims = reconcile_claims(claims) 
    ingest_claims(claims)

    print("Data inserted into Neo4j")