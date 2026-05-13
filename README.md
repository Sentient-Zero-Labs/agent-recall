# Recall

[![PyPI](https://img.shields.io/pypi/v/szl-recall)](https://pypi.org/project/szl-recall/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/szl-recall)](https://pypi.org/project/szl-recall/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Persistent memory layer for AI agents. Local-first. Inspectable. Framework-agnostic.

**PyPI:** [`szl-recall`](https://pypi.org/project/szl-recall/) &nbsp;·&nbsp; **GitHub:** [`Sentient-Zero-Labs/agent-recall`](https://github.com/Sentient-Zero-Labs/agent-recall)

Built as the implementation anchor for the [Building Effective Tools for AI](https://read.sentientzerolabs.com/tools/) series by Sentient Zero Labs.

---

## What Recall is

Recall is an MCP server that gives AI agents durable, structured memory across sessions.

An agent sends raw conversation text to Recall via `store_memory`. Recall extracts typed facts in the background using Claude Haiku — things like *"user prefers Python for backend services"* or *"team decided to use FastAPI for the new service"* — and stores them as searchable memories. When the agent needs context, it calls `search_memories` and gets back the most relevant memories ranked by a 4-component scoring model (relevance, recency, importance, access frequency).

Everything is local: one SQLite file, one process, no external services beyond the Anthropic API for extraction.

---

## Capabilities

**Memory types extracted from conversation text:**

| Type | Meaning | Example |
|---|---|---|
| `preference` | How the user likes to work | *"User prefers dark mode and vim keybindings"* |
| `fact` | Stated facts about the user or context | *"User works at Acme Corp on the payments team"* |
| `decision` | Choices made during conversation | *"Team decided to use Postgres over MySQL for this project"* |
| `procedure` | Processes the user described | *"User's deploy process: test → staging → manual approval → prod"* |

**Structured facts** — preferences and facts are also stored with `(entity, attribute, value)` triples, enabling deterministic contradiction detection. When a new memory contradicts an existing one (same entity + attribute, different value), the old memory is automatically superseded. Superseded memories are excluded from all queries.

**Hybrid search** — queries are ranked using BM25Plus (keyword relevance) fused with optional dense vector similarity via RRF (Reciprocal Rank Fusion, k=60), then re-scored with a 4-component formula:

```
score = w_rrf · RRF(BM25, cosine)
      + w_recency · exp(−age_days / (1 + access_count))
      + w_import  · importance
      + w_strength · log(1 + access_count) / log(1 + max_access_count)

Weights sum to 1.0 and shift linearly with recency_weight param:
  recency_weight=0 → (w_rrf=0.70, w_recency=0.00, w_import=0.20, w_strength=0.10)
  recency_weight=1 → (w_rrf=0.40, w_recency=0.40, w_import=0.10, w_strength=0.10)
```

**Two ingestion paths:**

1. **MCP tools** — async-acknowledge: `store_memory` returns in <10ms, extraction runs in a background worker.
2. **A2A protocol** — synchronous task interface for agent-to-agent calls. Supports a `consolidate_memories` skill with a contradiction resolution flow (`input-required` state when conflicts are detected).

**Security** — tool descriptions are validated at startup against injection patterns (URLs, conditional behavior instructions, exfiltration patterns). Bearer tokens are SHA-256 hashed before storage. A 30-second hard timeout prevents runaway tool calls.

---

## Install

```bash
pip install szl-recall
```

With optional dense vector search (requires ~500MB for the model on first run):

```bash
pip install "szl-recall[embeddings]"
```

---

## Quick start

```bash
# 1. Set your Anthropic API key (used for background extraction)
export ANTHROPIC_API_KEY=sk-ant-...

# 2. Start the server — this initializes the database on first run
recall serve --port 8000

# In a new terminal:

# 3. Create an API token for your user
recall create-token my-agent --db recall.db

#   Output:
#     Token created for namespace: my-agent
#
#       Bearer <token>
#
#     Store this token securely — it will not be shown again.
```

**Add to Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "recall": {
      "url": "http://localhost:8000/mcp",
      "transport": "streamable-http",
      "headers": {
        "Authorization": "Bearer <your-token>"
      }
    }
  }
}
```

Restart Claude Desktop. Recall's seven tools are now available in every conversation.

---

## Seven MCP tools

### `store_memory`

Stores conversation text for background extraction. Returns immediately (<10ms). Safe to retry with the same `idempotency_key`.

```
Arguments:
  text            str   — conversation transcript or fact to store
  topic           str   — context label (e.g. "engineering", "personal", "project-x")
  idempotency_key str   — unique key for this submission; duplicates are ignored
  session_id      str   — (optional) conversation ID for provenance
  agent_id        str   — (optional) which agent submitted this

Returns:
  {status: "ok", data: {queued: true, job_id: "..."}}     — new job
  {status: "ok", data: {queued: false, cached: true}}     — duplicate key
```

**What happens after**: the extraction worker picks up the job, calls Claude Haiku with the text, extracts 0–5 typed memories, checks for contradictions with existing facts, and persists to SQLite. The `job_id` can be used to correlate log entries.

### `search_memories`

Hybrid retrieval over all stored memories for the authenticated namespace.

```
Arguments:
  query          str     — natural language query
  limit          int     — max results (default 20, hard cap 50)
  recency_weight float   — 0.0–1.0 (default 0.3); higher surfaces newer memories
  mmr_lambda     float   — 0.0–1.0 (default 0.5); 0=max diversity, 1=pure relevance
  score_threshold float  — drop candidates below this hybrid score (default 0.0)
  max_tokens     int     — optional token budget; trims results to fit (4 chars/token)

Returns:
  {status: "ok", data: {results: [...memories], total: N}}
```

Each result includes: `id`, `text`, `topic`, `type`, `importance`, `created_at`.

**MMR diversification**: when `sentence-transformers` is installed, results are reranked using Max-Marginal Relevance so near-duplicate memories don't all crowd the top slots. Set `mmr_lambda=0.3` for high diversity, `mmr_lambda=1.0` to disable.

### `inspect_memories`

Paginated list of all active memories (superseded memories are excluded).

```
Arguments:
  limit   int   — page size (default 20, max 50)
  offset  int   — pagination offset (default 0)

Returns:
  {status: "ok", data: {memories: [...], total: N, has_more: bool, next_offset: N|null}}
```

### `delete_memory`

Permanently removes a memory. User-scoped — an agent can only delete its own memories.

```
Arguments:
  memory_id  str   — ID from inspect_memories or search_memories

Returns:
  {status: "ok", data: {deleted: "<id>"}}
  {status: "error", code: "MEMORY_NOT_FOUND"}
```

### `get_memory_stats`

Fast health check. Returns counts of active memories by type and the number of pending extraction jobs.

```
Arguments: none

Returns:
  {
    status: "ok",
    data: {
      by_type: {preference: N, fact: N, decision: N, procedure: N},
      total: N,
      pending_extractions: N
    }
  }
```

### `consolidate_memories`

Finds and merges semantically similar memories in a given topic using embedding-based clustering followed by an LLM merge pass. Reduces memory bloat and improves retrieval quality over time.

```
Arguments:
  topic                str    — topic to consolidate (e.g. "engineering")
  similarity_threshold float  — cosine similarity cutoff for clustering (default 0.85)
  dry_run              bool   — if true, returns a plan without modifying the database

Returns:
  {
    status: "ok",
    data: {
      groups_found: N,
      memories_consolidated: N,
      memories_created: N,
      deleted_ids: [...],
      created: [...],
      dry_run: bool
    }
  }
  {status: "error", code: "EMBEDDINGS_REQUIRED"}  — sentence-transformers not installed
```

**What happens**: active memories in the topic are embedded, clustered by cosine similarity, and each cluster is sent to Claude Haiku for a merge pass. Original memories are superseded (`valid_until` set); a single canonical memory is written in their place.

### `delete_namespace_data`

Permanently deletes ALL data for the authenticated namespace. Irreversible. Requires exact confirmation string.

```
Arguments:
  confirm  str  — must be exactly "DELETE MY DATA" to proceed

Returns:
  {status: "ok", data: {tokens_revoked: N}}
  {status: "error", code: "CONFIRM_REQUIRED"}
```

**What happens**: deletes all memories, operations, A2A tasks, and tool call records for the namespace. Revokes all API tokens. Subsequent requests with any token for this namespace will return 401.

---

## A2A protocol

Recall also implements the Agent-to-Agent (A2A) protocol for direct agent-to-agent calls. The agent card is published at `GET /.well-known/agent-card.json`.

**Create a consolidation task:**

```
POST /a2a
Authorization: Bearer <token>

{
  "skill": "consolidate_memories",
  "input": {
    "text": "Conversation transcript...",
    "topic": "engineering"
  }
}

→ 202 {id: "<task_id>", status: "submitted"}
```

**Poll for completion:**

```
GET /a2a/<task_id>

→ {id: "...", status: "working" | "completed" | "input-required", output: {...}}
```

**Resolve contradictions** (when `status == "input-required"`):

```
POST /a2a/<task_id>/resume

{
  "resolution": "keep_existing" | "keep_new" | "keep_both"
}
```

**A2A vs MCP store_memory:**

| | MCP `store_memory` | A2A `consolidate_memories` |
|---|---|---|
| Latency | <10ms (async-acknowledge) | 2–6s (synchronous extraction) |
| Contradiction handling | Auto-resolved via schema | `input-required` flow for ambiguous cases |
| Caller | Any MCP client | Another agent |

---

## Architecture

```
                      Agent / Claude Desktop
                             │
                    Bearer <token>
                             │
                             ▼
             ┌───────────────────────────────┐
             │       BearerAuthMiddleware     │  hash(token) → api_tokens
             │       TimeoutMiddleware        │  asyncio.wait_for(30s)
             │       LoggingMiddleware        │  ToolCallRecord per call
             └───────────────┬───────────────┘
                             │
               ┌─────────────┴──────────────┐
               │                            │
               ▼                            ▼
        POST /mcp                       POST /a2a
        FastMCP tools                   A2A task router
               │                            │
               │  store_memory              │  consolidate_memories
               │    ├── INSERT operations   │    ├── worker.extract() [sync]
               │    ├── enqueue job ──────► │    ├── detect contradictions
               │    └── return <10ms        │    └── bulk_store / input-required
               │                            │
               │  search_memories           GET /a2a/<id>   poll
               │    └── _hybrid_search      POST /a2a/<id>/resume
               │         ├── BM25Plus ranks
               │         ├── dense ranks (optional)
               │         ├── RRF fusion
               │         └── 4-component score
               │
               ▼
         ┌──────────────┐     asyncio.Queue
         │   SQLite DB  │ ◄──────────────────── ExtractionWorker
         │   (WAL mode) │                          │
         └──────────────┘                   Claude Haiku API
                                            extract → structured facts
                                            → _handle_contradiction
                                            → INSERT memories
```

**Key invariants:**

- Every DB operation is scoped to `namespace` — injected via `ContextVar` from auth middleware, never passed as a tool argument. A namespace can represent a user, agent, project, or any string identity.
- `store_memory` is idempotent: `INSERT OR IGNORE` + `rowcount` check makes concurrent retries safe.
- Active memories are always `WHERE valid_until IS NULL`. Superseded facts remain in the DB (for audit) but are excluded from all queries.
- On server restart, `_recover_orphaned_operations()` marks any stuck jobs as `failed` to prevent ghost jobs.

---

## Database schema

Six tables in `recall.db`:

| Table | Purpose |
|---|---|
| `memories` | Core store. Active rows have `valid_until IS NULL`. |
| `operations` | Idempotency + job lifecycle tracking for async extraction. |
| `api_tokens` | SHA-256 hashed bearer tokens per namespace. |
| `tool_call_records` | Structured audit log for every tool call (duration, tokens, cost). |
| `schema_version` | Migration tracking. |
| `a2a_tasks` | Persistent A2A task store. Survives server restart. |

**Key `memories` columns:**

| Column | Type | Description |
|---|---|---|
| `text` | TEXT | The memory content |
| `type` | TEXT | `preference`, `fact`, `decision`, `procedure` |
| `entity` | TEXT | Subject of structured fact (e.g. `"user"`) |
| `attribute` | TEXT | Property (e.g. `"preferred_language"`) |
| `value` | TEXT | Value (e.g. `"Python"`) |
| `valid_from` | TEXT | When this fact became true |
| `valid_until` | TEXT | When superseded (NULL = still active) |
| `importance` | REAL | 0.0–1.0, LLM-assigned |
| `access_count` | INT | Increments on retrieval (used in recency decay) |
| `embedding` | BLOB | L2-normalized float32 vector (BAAI/bge-small-en-v1.5) |

---

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | — | Claude Haiku for background extraction |
| `RECALL_DB_PATH` | No | `recall.db` | Path to SQLite database file |
| `RECALL_DB_URL` | No | — | Postgres DSN (e.g. `postgresql://user:pass@localhost/recall`). If set, uses Postgres instead of SQLite. |
| `RECALL_DECAY_LAMBDA` | No | `0.02` | Decay rate — ~35-day half-life at default. |
| `RECALL_DECAY_JOB_INTERVAL` | No | `3600` | Seconds between decay runs. |

---

## Development

```bash
git clone https://github.com/Sentient-Zero-Labs/agent-recall
cd recall

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run all tests (worker tests require ANTHROPIC_API_KEY)
pytest tests/ -v

# Fast tests only — no API key needed (~2s)
pytest tests/test_search.py tests/test_server.py tests/test_client.py tests/test_models.py -v

# With embeddings (requires sentence-transformers)
pip install -e ".[dev,embeddings]"
pytest tests/test_mmr.py -v   # MMR deduplication + diversity tests

# Live extraction tests (~8s)
ANTHROPIC_API_KEY=sk-ant-... pytest tests/test_worker.py -v -s
```

**Test coverage by layer:**

| File | What it covers |
|---|---|
| `test_client.py` | SQLite layer — store, get, search, delete, pagination |
| `test_models.py` | MemoryUnit validation and serialization |
| `test_server.py` | HTTP layer — auth, MCP routing, response shapes, error codes |
| `test_search.py` | Search correctness (superseded filtering, idempotency) + ranking behavior |
| `test_worker.py` | Extraction pipeline — Haiku output structure, queue→DB path, stub fallback |
| `test_mmr.py` | MMR reranking, diversity, score_threshold, fallback |
| `test_budget.py` | Context budget trimming (`max_tokens`) |
| `test_gdpr.py` | `delete_namespace_data` — wrong confirm, full erasure, token revocation |
| `test_postgres_backend.py` | DB backend abstraction, placeholder translation, factory |
| `test_decay.py` | DecayWorker — decay scoring, access-count protection, run_once count |
| `test_consolidation.py` | `consolidate_memories` — clustering, LLM merge, dry-run mode |

---

## CLI reference

```bash
recall serve [--host 0.0.0.0] [--port 8000] [--db recall.db] [--reload]
recall create-token <namespace> [--db recall.db]
recall status [--db recall.db]
```

`<namespace>` can be any string: `alice`, `agent:code-reviewer-v2`, `project:payments`, etc.

Note: `--db` flag on `recall serve` now correctly overrides `RECALL_DB_PATH` env var.

---

## Version history

- **v0.3.3** (current): `user_id` renamed to `namespace` — tokens, DB columns, and all queries. Namespace can be a user, agent (`agent:code-reviewer-v2`), project, or any string. DB migration runs automatically on startup. MMR bug fix: relevance signal now uses hybrid scores (recency + decay + BM25 + density) instead of raw cosine, so `mmr_lambda=1.0` correctly preserves full ranking order.
- **v0.3.0–v0.3.2**: MMR diversification (`mmr_lambda` param), context budget trimming (`max_tokens`), GDPR erasure (`delete_namespace_data`, 7th tool), A2A task persistence (survives restarts), Postgres backend (`RECALL_DB_URL`), `--db` CLI bug fix, TROUBLESHOOTING.md, sentence-transformers 5.5.0 validated.
- **v0.2**: Hybrid BM25Plus+RRF search, 4-component scoring, structured fact extraction (entity/attribute/value), deterministic contradiction detection, A2A consolidation worker.
- **v0.1**: SQLite + BM25 search, 5 MCP tools, async extraction via Claude Haiku, bearer auth.

**Roadmap:**
- Multi-namespace admin API (list/delete namespaces)
- Webhook notifications on extraction completion
- `owner_id + agent_id` composite isolation for agent-per-user memory separation (v0.4)

---

## License

MIT — see [LICENSE](LICENSE).

---

Built by [Sentient Zero Labs](https://sentientzerolabs.com).
Newsletter: [read.sentientzerolabs.com](https://read.sentientzerolabs.com).
