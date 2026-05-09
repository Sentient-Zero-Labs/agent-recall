# Changelog

All notable changes to Recall are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Planned (v0.2 — Memory series)
- LLM-based extraction pipeline (structured fact extraction from conversation transcripts)
- Vector embeddings via `sentence-transformers` (all-MiniLM-L6-v2)
- Hybrid search: BM25 + cosine similarity merged with RRF (k=60)
- Contradiction detection and `superseded_by` resolution
- Decay scoring for memory relevance over time
- Postgres + pgvector support (optional `postgres` dependency group)
- `get_job_status` tool for checking extraction progress

---

## [0.1.0] — 2026-05

### Added
- `MemoryUnit` dataclass with `MemoryType` enum (preference, fact, decision, procedure)
- SQLite schema: `memories`, `tool_call_records`, `operations`, `schema_version`
- Five MCP tools via FastMCP: `store_memory`, `search_memories`, `inspect_memories`,
  `delete_memory`, `get_memory_stats`
- `BearerAuthMiddleware` — bearer token validation with ContextVar injection
- `TimeoutMiddleware` — 30s hard cap on all tool calls via `asyncio.wait_for`
- `ExtractionWorker` — async queue worker for background memory processing (v0.1 stub)
- `MemoryClient` — direct SQLite access for tests and local use
- `ToolCallRecord` dataclass + `LoggingMiddleware` for per-call observability
- Tool description security validation at startup (poisoning pattern detection)
- `recall serve` and `recall status` CLI commands
- WAL mode SQLite configuration
- BM25 keyword search via `rank-bm25`
- Test suite: `test_models.py`, `test_client.py` with pytest-asyncio

### Architecture notes
- `store_memory` uses async-acknowledge pattern: returns in <10ms, extraction runs async
- All write tools accept `idempotency_key` — safe to retry on network failure
- All list tools have default limit (20) and hard cap (50) — no unbounded queries
- All error returns are typed dicts: `{status, error, code}` — LLM-readable
