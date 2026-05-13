# Changelog

All notable changes to Recall are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [0.3.4] — 2026-05

### Changed
- `delete_user_data` tool renamed to `delete_namespace_data` to match the `namespace` terminology used everywhere else.
- `store_memory`: `idempotency_key` parameter is now optional (default `""`). When omitted, a UUID is auto-generated so callers don't need to supply one.
- `__version__` in `recall/__init__.py` updated to match `pyproject.toml`.

---

## [0.3.3] — 2026-05

### Changed
- **`user_id` renamed to `namespace`** across all DB columns, Python code, and CLI. Existing DBs are migrated automatically on startup via `ALTER TABLE ... RENAME COLUMN`. No data loss.
- `recall create-token` positional arg is now named `namespace` in help text and output
- MMR reranking: relevance signal now uses normalised hybrid scores (recency + decay + BM25 + density) instead of raw cosine similarity. At `mmr_lambda=1.0`, output order now correctly matches hybrid ranking order.

### Fixed
- MMR was skipped when `len(candidates) == limit` (the most common case). Guard condition corrected — MMR now always runs when embeddings are available.
- `recall serve --db <path>` now correctly overrides `RECALL_DB_PATH`. Previously the `--db` flag was silently ignored due to a module-level import race.

---

## [0.3.0] — 2026-05

### Added
- **MMR diversification** (`mmr_lambda` param on `search_memories`): post-ranking reorder using Max-Marginal Relevance. Set `mmr_lambda=0.3` for high diversity, `1.0` for pure relevance. Requires `sentence-transformers` for full effect; falls back to BM25 order if model unavailable.
- **Context budget trimming** (`max_tokens` param on `search_memories`): greedy trim of results to fit a token budget (4 chars/token heuristic). Always includes at least 1 result.
- **Score threshold filtering** (`score_threshold` param on `search_memories`): drop candidates below a hybrid score floor before MMR.
- **`delete_namespace_data` MCP tool** (7th tool): GDPR-style erasure. Deletes all memories, operations, A2A tasks, and tool call records for the namespace. Revokes all tokens. Requires `confirm="DELETE MY DATA"`.
- **A2A task persistence**: `a2a_tasks` table stores task state in SQLite. Tasks survive server restarts. In-memory `_tasks` dict is now a write-through cache.
- **Postgres backend**: set `RECALL_DB_URL=postgresql://...` to use asyncpg + connection pooling instead of SQLite. Automatic `?` → `$1, $2, ...` placeholder translation. SQLite remains the default.
- `TROUBLESHOOTING.md` covering 5 common deployment issues.
- `sentence-transformers` 5.5.0 validated; all MMR tests pass with embeddings installed.
- 92-test suite (was 63 in v0.2).

### Architecture
- `_migrate_v3()` adds `a2a_tasks` table to existing DBs on startup.
- `DatabaseBackend` ABC with `SQLiteBackend` and `PostgresBackend` implementations.
- Factory `get_backend()` selects backend from `RECALL_DB_URL` env var.

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
