# Recall

Persistent memory layer for AI agents. Local-first. Inspectable. Framework-agnostic.

Built as the implementation anchor for the [Building Effective Tools for AI](https://read.sentientzerolabs.com) series by Sentient Zero Labs.

---

## What it does

Recall gives AI agents persistent, structured memory across sessions.

- **Store** conversation transcripts → extracted as typed facts, preferences, decisions, procedures
- **Search** memories using hybrid retrieval (BM25 + vector in v0.2)
- **Inspect** stored memories with pagination
- **Delete** specific memories
- **Observe** every operation via structured `ToolCallRecord` logging

## Install

```bash
pip install recall-memory
```

With optional vector search support:

```bash
pip install "recall-memory[embeddings]"
```

With Postgres backend:

```bash
pip install "recall-memory[postgres]"
```

## Quick start

```bash
# Start the MCP server
recall serve --host 0.0.0.0 --port 8000

# Check status
recall status
```

Add to your MCP client (Claude Desktop example):

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

## Five tools

| Tool | Purpose | Pattern |
|------|---------|---------|
| `store_memory` | Store conversation for extraction | Async-acknowledge, idempotent |
| `search_memories` | Hybrid retrieval by query | Synchronous, paginated |
| `inspect_memories` | List all memories | Synchronous, paginated |
| `delete_memory` | Remove one memory by ID | Synchronous |
| `get_memory_stats` | Counts + pending queue | Synchronous, fast health check |

## Architecture

```
Agent
  │
  │  MCP (Streamable HTTP)
  │
  ▼
┌──────────────────────┐
│  BearerAuthMiddleware │  ← validates token, injects user_id via ContextVar
│  TimeoutMiddleware    │  ← 30s hard cap via asyncio.wait_for
│  LoggingMiddleware    │  ← ToolCallRecord per call
├──────────────────────┤
│  store_memory         │  ← enqueue + return fast (<10ms)
│  search_memories      │  ← BM25 search (v0.1), +vectors (v0.2)
│  inspect_memories     │  ← paginated list
│  delete_memory        │  ← ownership-checked delete
│  get_memory_stats     │  ← health check
└──────────────────────┘
  │                   │
  │ SQLite (WAL mode)  │  asyncio.Queue
  │                   ▼
  │         ┌──────────────────┐
  │         │ ExtractionWorker │  ← processes store_memory jobs async
  │         └──────────────────┘
  ▼
recall.db
```

## Development

```bash
git clone https://github.com/ppritish51/recall
cd recall

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest tests/
```

## Versions

- **v0.1** (current): SQLite + BM25 search + 5 MCP tools. Extraction is stubbed — stores raw text.
- **v0.2** (Memory series): Full LLM extraction, vector embeddings, hybrid RRF search, Postgres support.

## License

MIT — see [LICENSE](LICENSE).

---

Built by [Sentient Zero Labs](https://sentientzerolabs.com).
Newsletter: [read.sentientzerolabs.com](https://read.sentientzerolabs.com).
