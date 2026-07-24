# SOW Agentic Memory System — Phase 0 (Setup)

## Folder structure (will grow phase by phase)

```
sow-agent/
├── infra/
│   └── docker-compose.yml      # Kafka, Zookeeper, Kafka UI, Redis, Neo4j, Postgres
├── src/
│   ├── ingestion/               # Phase 2 — Kafka producers/consumers, DLQ
│   ├── extraction/               # Phase 3 — LLM extraction workers, review queue
│   ├── memory/                   # Phase 4 — FAISS + Neo4j read/write, reconciliation
│   ├── cache/                    # Phase 5 — Redis warm cache + rate limiter
│   ├── agent/                    # Phase 6 — LangGraph nodes, MCP tool wiring
│   ├── api/                      # Phase 9 — FastAPI app
│   └── schemas/                  # Phase 1 — Pydantic models, Kafka message schemas
├── data/
│   ├── faiss_index/               # local FAISS index files
│   └── sample_sows/                # your example SOW documents go here
├── docs/                          # ontology docs, ADRs, diagrams
├── requirements.txt
├── .env.example
└── README.md
```

## One-time setup

1. **Install Docker & Docker Compose** (if not already installed).
2. **Python environment** (3.11+ recommended):
   ```bash
   cd sow-agent
   python3 -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Copy env template and fill in your keys:**
   ```bash
   cp .env.example .env
   # edit .env: add OPENAI_API_KEY, HUGGINGFACE_API_KEY
   # LANGSMITH_API_KEY optional for now — needed from Phase 7 onward
   ```
4. **Bring up infrastructure:**
   ```bash
   cd infra
   docker compose up -d
   ```
5. **Verify each service is reachable:**

   | Service | Check |
   |---|---|
   | Kafka UI | http://localhost:8080 — should show cluster `sow-local` |
   | Neo4j Browser | http://localhost:7474 — login `neo4j` / `sow_password_change_me` |
   | Redis | `docker exec -it sow-redis redis-cli ping` → `PONG` |
   | Postgres | `docker exec -it sow-postgres psql -U sow_admin -d sow_agent -c "\dt"` |

6. **Drop your sample SOW documents** into `data/sample_sows/` — we'll use these in Phase 1 (ontology design) to reverse-engineer what fields/entities actually show up in real SOWs, and again in Phase 3 (extraction) as test input.

## Notes on configuration choices

- `KAFKA_AUTO_CREATE_TOPICS_ENABLE=true` is dev-only. In Phase 2 we switch to explicit topic creation (fixed partition counts, retention policy) because auto-created topics get default configs that are wrong for production (e.g. 1 partition, infinite retention).
- Neo4j memory settings (`pagecache=512M`, `heap=1G`) are safe local-dev defaults. We'll revisit sizing once we know real graph volume.
- All passwords in `docker-compose.yml` are placeholders — rotate before anything resembling production.
- Postgres is the addition to your original stack (see explanation in chat) — it's the system-of-record for review queue status and SOW version history; Kafka/Neo4j/FAISS are not good fits for "give me all SOWs pending my review."

## Status: Phase 0 complete when...

- [ ] `docker compose up -d` runs clean, all 6 containers healthy
- [ ] You can open Kafka UI, Neo4j Browser
- [ ] `pip install -r requirements.txt` succeeds
- [ ] `.env` has real OpenAI + HuggingFace keys
- [ ] Sample SOW docs dropped into `data/sample_sows/`
