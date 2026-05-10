"""Recall MCP server — all 5 tools, auth middleware, 30s timeout."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import aiosqlite
from fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from recall.db.connection import get_db_path, init_db
from recall.logging import LoggingMiddleware
from recall.security import hash_token, validate_tool_descriptions
from recall.worker import ExtractionWorker

logger = logging.getLogger(__name__)

user_id_ctx: ContextVar[str] = ContextVar("user_id", default="")
extraction_worker: ExtractionWorker | None = None


# ── Middleware ────────────────────────────────────────────────────────────────

class BearerAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Any) -> Any:
        # Agent Card is publicly discoverable — no auth required
        if request.url.path.startswith("/.well-known"):
            return await call_next(request)
        auth = request.headers.get("Authorization", "")
        token = auth.removeprefix("Bearer ").strip()
        user_id = await _validate_token(token)
        if not user_id:
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        user_id_ctx.set(user_id)
        return await call_next(request)


class TimeoutMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Any) -> Any:
        try:
            return await asyncio.wait_for(call_next(request), timeout=30.0)
        except asyncio.TimeoutError:
            return JSONResponse(
                {
                    "status": "error",
                    "error": "Tool call exceeded 30s limit.",
                    "code": "TOOL_TIMEOUT",
                },
                status_code=504,
            )


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(server: FastMCP) -> AsyncIterator[None]:
    global extraction_worker
    await init_db()
    _validate_server_descriptions()
    extraction_worker = ExtractionWorker()
    await extraction_worker.start()
    await _recover_orphaned_operations()
    yield
    if extraction_worker:
        await extraction_worker.stop()


async def _recover_orphaned_operations() -> None:
    """Mark any queued/processing operations left over from a prior crash as failed."""
    async with aiosqlite.connect(get_db_path()) as db:
        cursor = await db.execute(
            "UPDATE operations SET status = 'failed', updated_at = datetime('now') "
            "WHERE status IN ('queued', 'processing')"
        )
        await db.commit()
    if cursor.rowcount:
        logger.warning(
            "startup_orphan_recovery",
            extra={"count": cursor.rowcount},
        )


def _validate_server_descriptions() -> None:
    """Validate all tool descriptions against poisoning patterns. Fail fast."""
    tool_functions = [
        store_memory, search_memories, inspect_memories,
        delete_memory, get_memory_stats,
    ]
    descriptions = {
        f.__name__: (f.__doc__ or "").split("\n")[0].strip()
        for f in tool_functions
    }
    validate_tool_descriptions(descriptions)


# ── MCP server ────────────────────────────────────────────────────────────────

mcp = FastMCP("recall", lifespan=_lifespan)


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
async def store_memory(
    text: str,
    topic: str,
    idempotency_key: str,
    session_id: str = "",
    agent_id: str = "",
) -> dict:
    """Store conversation messages for background memory extraction.
    Returns immediately — extraction runs async. Safe to retry with the same key."""
    user_id = user_id_ctx.get()

    async with aiosqlite.connect(get_db_path()) as db:
        existing = await db.execute_fetchall(
            "SELECT 1 FROM operations WHERE idempotency_key = ?",
            (idempotency_key,),
        )
        if existing:
            return {"status": "ok", "data": {"queued": False, "cached": True}, "error": None}

        job_id = str(uuid.uuid4())
        await db.execute(
            "INSERT INTO operations (id, idempotency_key, user_id, status) VALUES (?,?,?,'queued')",
            (job_id, idempotency_key, user_id),
        )
        await db.commit()

    if extraction_worker:
        await extraction_worker.enqueue({
            "job_id": job_id,
            "user_id": user_id,
            "text": text,
            "topic": topic,
            "session_id": session_id,
            "agent_id": agent_id,
        })

    return {"status": "ok", "data": {"queued": True, "job_id": job_id}, "error": None}


@mcp.tool()
async def search_memories(
    query: str,
    limit: int = 20,
    recency_weight: float = 0.3,
) -> dict:
    """Search memories using hybrid retrieval (~200-500ms). recency_weight 0-1 upweights recent results."""
    if not 0.0 <= recency_weight <= 1.0:
        return {
            "status": "error",
            "error": "recency_weight must be between 0.0 and 1.0.",
            "code": "INVALID_PARAM",
        }
    user_id = user_id_ctx.get()
    results = await _hybrid_search(user_id, query, min(limit, 50), recency_weight)
    return {"status": "ok", "data": {"results": results, "total": len(results)}, "error": None}


@mcp.tool()
async def inspect_memories(limit: int = 20, offset: int = 0) -> dict:
    """List stored memories with pagination. Default 20 per page, max 50."""
    limit = min(limit, 50)
    user_id = user_id_ctx.get()

    async with aiosqlite.connect(get_db_path()) as db:
        rows = await db.execute_fetchall(
            "SELECT id, text, topic, importance, type, created_at "
            "FROM memories WHERE user_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (user_id, limit, offset),
        )
        count = await db.execute_fetchall(
            "SELECT COUNT(*) FROM memories WHERE user_id = ?", (user_id,)
        )
        total = count[0][0]

    memories = [
        {
            "id": r[0],
            "text": r[1],
            "topic": r[2],
            "importance": r[3],
            "type": r[4],
            "created_at": r[5],
        }
        for r in rows
    ]
    return {
        "status": "ok",
        "data": {
            "memories": memories,
            "total": total,
            "has_more": offset + limit < total,
            "next_offset": offset + limit if offset + limit < total else None,
        },
        "error": None,
    }


@mcp.tool()
async def delete_memory(memory_id: str) -> dict:
    """Permanently delete a memory by ID. This action cannot be undone."""
    user_id = user_id_ctx.get()

    async with aiosqlite.connect(get_db_path()) as db:
        cursor = await db.execute(
            "DELETE FROM memories WHERE id = ? AND user_id = ?",
            (memory_id, user_id),
        )
        await db.commit()
        if cursor.rowcount == 0:
            return {
                "status": "error",
                "error": (
                    f"Memory '{memory_id}' not found. "
                    "Use inspect_memories to list valid IDs."
                ),
                "code": "MEMORY_NOT_FOUND",
            }

    return {"status": "ok", "data": {"deleted": memory_id}, "error": None}


@mcp.tool()
async def get_memory_stats() -> dict:
    """Return memory counts and storage stats for the current user. Fast health check."""
    user_id = user_id_ctx.get()

    async with aiosqlite.connect(get_db_path()) as db:
        by_type = await db.execute_fetchall(
            "SELECT type, COUNT(*) FROM memories WHERE user_id = ? GROUP BY type",
            (user_id,),
        )
        pending = await db.execute_fetchall(
            "SELECT COUNT(*) FROM operations WHERE user_id = ? AND status = 'queued'",
            (user_id,),
        )

    return {
        "status": "ok",
        "data": {
            "by_type": dict(by_type),
            "total": sum(r[1] for r in by_type),
            "pending_extractions": pending[0][0],
        },
        "error": None,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

async def _validate_token(token: str) -> str | None:
    if not token:
        return None
    token_hash = hash_token(token)
    async with aiosqlite.connect(get_db_path()) as db:
        rows = await db.execute_fetchall(
            "SELECT user_id FROM api_tokens WHERE token_hash = ? AND revoked = 0",
            (token_hash,),
        )
    return rows[0][0] if rows else None


def _parse_ts(ts: str) -> datetime:
    """Parse ISO8601 timestamp from SQLite, ensuring UTC timezone."""
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _bm25_ranks(query: str, texts: list[str]) -> list[int] | None:
    """Return per-document BM25 rank (1-based). None if rank_bm25 not installed.

    Uses BM25Plus over BM25Okapi: BM25Okapi IDF collapses to 0 with N=2 when a term
    appears in exactly one document (log(1.5/1.5)=0), causing all scores to be 0.
    BM25Plus adds a lower-bound delta that keeps scores positive for small corpora.
    """
    try:
        from rank_bm25 import BM25Plus
        tokenized = [t.lower().split() for t in texts]
        bm25 = BM25Plus(tokenized)
        scores = list(bm25.get_scores(query.lower().split()))
        # rank = position in descending score order (0-based → 1-based)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        ranks = [0] * len(scores)
        for rank, idx in enumerate(order):
            ranks[idx] = rank + 1  # 1-based
        # Docs with zero BM25Plus score have no query term overlap — push to worst rank
        for i, s in enumerate(scores):
            if s == 0:
                ranks[i] = len(scores) + 1
        return ranks
    except ImportError:
        return None


def _dense_ranks(query: str, texts: list[str]) -> list[int] | None:
    """Return per-document cosine similarity rank (1-based). None if model unavailable."""
    from recall.embeddings import embed, embed_query
    q_vec = embed_query(query)
    if q_vec is None:
        return None
    doc_vecs = embed(texts)
    if doc_vecs is None:
        return None
    from recall.embeddings import cosine_scores
    sims = cosine_scores(q_vec, doc_vecs)
    order = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
    ranks = [0] * len(sims)
    for rank, idx in enumerate(order):
        ranks[idx] = rank + 1
    return ranks


def _rrf_fuse(
    bm25_ranks: list[int] | None,
    dense_ranks: list[int] | None,
    k: int = 60,
) -> list[float]:
    """Reciprocal Rank Fusion. Returns a score per document (higher = better match)."""
    n = len(bm25_ranks or dense_ranks or [])
    if n == 0:
        return []
    worst = n + 1
    scores = [0.0] * n
    for ranks in (bm25_ranks, dense_ranks):
        if ranks is None:
            continue
        for i, r in enumerate(ranks):
            if r <= n:  # skip placeholder worst-rank docs
                scores[i] += 1.0 / (k + r)
    return scores


async def _hybrid_search(
    user_id: str, query: str, limit: int, recency_weight: float = 0.3
) -> list[dict]:
    """BM25 + dense RRF(k=60) hybrid retrieval with 4-component scoring.

    Components (Park et al. 2023 + MemoryBank):
      0.50 · RRF(BM25, cosine)
      0.25 · exp(-Δt / (1 + access_count))   — recency with strength modifier
      0.15 · importance
      0.10 · log(1+ac) / log(1+max_ac)       — access frequency (strength)
    """
    async with aiosqlite.connect(get_db_path()) as db:
        rows = await db.execute_fetchall(
            "SELECT id, text, topic, importance, type, created_at, access_count "
            "FROM memories WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit * 5),
        )

    if not rows:
        return []

    memories = [
        {
            "id": r[0], "text": r[1], "topic": r[2], "importance": r[3],
            "type": r[4], "created_at": r[5], "access_count": r[6] or 0,
        }
        for r in rows
    ]
    texts = [m["text"] for m in memories]

    bm25_r = _bm25_ranks(query, texts)
    dense_r = _dense_ranks(query, texts)
    rrf = _rrf_fuse(bm25_r, dense_r, k=60)

    now = datetime.now(timezone.utc)
    max_ac = max((m["access_count"] for m in memories), default=1) or 1

    scored: list[tuple[float, dict]] = []
    for i, m in enumerate(memories):
        if rrf[i] == 0:
            continue  # no BM25 or dense signal — suppress from results
        age_days = (now - _parse_ts(m["created_at"])).total_seconds() / 86400
        ac = m["access_count"]
        recency = math.exp(-age_days / (1 + ac))
        strength = math.log(1 + ac) / math.log(1 + max_ac)
        # Scale recency coefficient by user's recency_weight param (baseline = 0.3)
        recency_coeff = 0.25 * (recency_weight / 0.3) if recency_weight > 0 else 0.0
        score = (
            0.50 * rrf[i]
            + recency_coeff * recency
            + 0.15 * (m.get("importance") or 0.5)
            + 0.10 * strength
        )
        scored.append((score, m))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:limit]]


# ── ASGI app factory ──────────────────────────────────────────────────────────

def create_app():
    """Return the Starlette app with all middleware and A2A routes wired. Used by uvicorn."""
    from recall.a2a import create_a2a_router, create_well_known_router

    app = mcp.http_app(transport="streamable-http")
    app.mount("/a2a", create_a2a_router(user_id_ctx))
    app.mount("/.well-known", create_well_known_router())
    # Starlette applies middleware in reverse order — LoggingMiddleware runs outermost (sees full duration)
    app.add_middleware(LoggingMiddleware, user_id_ctx=user_id_ctx)
    app.add_middleware(TimeoutMiddleware)
    app.add_middleware(BearerAuthMiddleware)
    return app
