Claim proposal

class MemoryNode:
    fact_id: str
    content: dict          # normalized fact
    confidence: float      # 0-1, Bayesian updated
    valid_from: datetime
    valid_until: datetime | None   # None = still believed true
    superseded_by: str | None      # links to newer fact_id
    source_sessions: list[str]     # provenance
    observation_count: int         # how many times confirmed