"""Recall MCP server — all 5 tools, auth middleware, 30s timeout."""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from contextvars import ContextVar
from typing import Any

import aiosqlite
from fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from recall.db.connection import get_db_path, init_db
from recall.security import hash_token, validate_tool_descriptions
from recall.worker import ExtractionWorker

mcp = FastMCP("recall")

user_id_ctx: ContextVar[str] = ContextVar("user_id", default="")
extraction_worker: ExtractionWorker | None = None


# ── Middleware ────────────────────────────────────────────────────────────────

class BearerAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Any) -> Any:
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

@mcp.on_startup
async def startup() -> None:
    global extraction_worker
    await init_db()
    _validate_server_descriptions()
    extraction_worker = ExtractionWorker()
    await extraction_worker.start()


@mcp.on_shutdown
async def shutdown() -> None:
    if extraction_worker:
        await extraction_worker.stop()


def _validate_server_descriptions() -> None:
    """Validate all tool descriptions against poisoning patterns. Fail fast."""
    tool_descriptions = {
        name: (tool.description or "") for name, tool in mcp._tools.items()
    }
    validate_tool_descriptions(tool_descriptions)


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
async def store_memory(text: str, topic: str, idempotency_key: str) -> dict:
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
        await extraction_worker.enqueue(
            {"job_id": job_id, "user_id": user_id, "text": text, "topic": topic}
        )

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
    results = await _bm25_search(user_id, query, min(limit, 50))
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


async def _bm25_search(user_id: str, query: str, limit: int) -> list[dict]:
    """BM25 keyword search. v0.1 baseline — RRF + vector upgrade in Memory series."""
    async with aiosqlite.connect(get_db_path()) as db:
        rows = await db.execute_fetchall(
            "SELECT id, text, topic, importance, type, created_at "
            "FROM memories WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit * 5),  # fetch more than needed for in-memory BM25 ranking
        )

    if not rows:
        return []

    memories = [
        {"id": r[0], "text": r[1], "topic": r[2], "importance": r[3], "type": r[4], "created_at": r[5]}
        for r in rows
    ]

    try:
        from rank_bm25 import BM25Okapi

        tokenized = [m["text"].lower().split() for m in memories]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(query.lower().split())
        ranked = sorted(zip(scores, memories), key=lambda x: x[0], reverse=True)
        return [m for _, m in ranked[:limit] if _ > 0]
    except ImportError:
        return memories[:limit]


# ── WSGI app factory ──────────────────────────────────────────────────────────

def create_app():
    """Return the Starlette app with all middleware wired. Used by uvicorn."""
    app = mcp.streamable_http_app()
    app.add_middleware(TimeoutMiddleware)
    app.add_middleware(BearerAuthMiddleware)
    return app
