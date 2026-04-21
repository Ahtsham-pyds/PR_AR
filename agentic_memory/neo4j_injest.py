from neo4j import GraphDatabase
from reconciliation import reconcile_claims

URI = "neo4j://127.0.0.1:7687"
USER = "neo4j"
PASSWORD = "password"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))


import re

def normalize_predicate_for_neo4j(p):
    p = p.strip().upper()
    p = re.sub(r'[^A-Z0-9_]', '_', p)  # replace invalid chars
    p = re.sub(r'_+', '_', p)          # collapse multiple underscores
    return p

# -----------------------
# HELPER: CREATE ENTITY
# -----------------------
def create_entity(tx, name):
    tx.run("""
        MERGE (e:Entity {name: $name})
    """, name=name)
    return None

# -----------------------
# HELPER: CREATE SOW NODE
# -----------------------
def create_sow(tx, sow_id):
    tx.run("""
        MERGE (s:SOW {id: $sow_id})
    """, sow_id=sow_id)
    
    return None


# -----------------------
# CREATE RELATIONSHIP
# -----------------------
# def create_relationship(tx, sow_id, predicate, obj, confidence,sources):
#     predicate = normalize_predicate_for_neo4j(predicate)
#     query = f"""
#     MATCH (s:SOW {{id: $sow_id}})
#     MATCH (e:Entity {{name: $obj}})
#     MERGE (s)-[r:{predicate}]->(e)
#     SET r.confidence = $confidence,
#         r.sources = $sources
#     """
#     tx.run(
#     query,
#     sow_id=sow_id,
#     obj=obj,
#     confidence=confidence,
#     sources=sources
# )
#     return None


SINGLE_VALUED = {
    "HAS_DURATION",
    "HAS_CONTRACT_TYPE",
    "USES_CURRENCY",
    "PREFERRED_VENDOR"
}


def create_relationship(tx, sow_id, predicate, obj, confidence,sources):
    predicate = normalize_predicate_for_neo4j(predicate)
    # Ensure nodes exist
    tx.run("""
        MERGE (s:SOW {id: $sow_id})
    """, sow_id=sow_id)

    tx.run("""
        MERGE (e:Entity {name: $obj})
    """, obj=obj)

    if predicate in SINGLE_VALUED:
        # 🔴 Step 1: Deactivate existing active relationship
        tx.run(f"""
        MATCH (s:SOW {{id: $sow_id}})-[r:{predicate}]->()
        WHERE r.is_active = true
        SET r.is_active = false,
            r.valid_to = datetime()
        """, sow_id=sow_id)

        # 🟢 Step 2: CREATE new relationship (THIS is where your line goes)
        tx.run(f"""
        MATCH (s:SOW {{id: $sow_id}})
        MATCH (e:Entity {{name: $obj}})
        CREATE (s)-[r:{predicate} {{
            valid_from: datetime(),
            is_active: true,
            confidence: $confidence
        }}]->(e)
        """, sow_id=sow_id, obj=obj, confidence=confidence)

    else:
        # Multi-valued → allow multiple
        tx.run(f"""
        MATCH (s:SOW {{id: $sow_id}})
        MATCH (e:Entity {{name: $obj}})
        MERGE (s)-[r:{predicate}]->(e)
        SET r.confidence = $confidence,
        r.sources = $sources
        """, sow_id=sow_id, obj=obj, confidence=confidence,sources=sources)


# -----------------------
# MAIN INGEST FUNCTION
# -----------------------
def ingest_claims(claims):
    with driver.session(database="sow1") as session:
        for c in claims:
            sow_id = c["subject"]
            predicate = c["predicate"]
            obj = c["object"]
            confidence = c["confidence"]
            sources = c["sources"]

            session.execute_write(create_sow, sow_id)
            session.execute_write(create_entity, obj)
            session.execute_write(create_relationship, sow_id, predicate, obj, confidence,sources)
        return 'Claims ingested successfully'


# -----------------------
# RUN
# -----------------------
# if __name__ == "__main__":
#     from extraction import run_extraction

#     claims = run_extraction()
#     claims = reconcile_claims(claims) 
#     ingest_claims(claims)

