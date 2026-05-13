"""A2A consolidation worker — exposes Recall as an A2A agent.

Agent Card: GET /.well-known/agent-card.json
Task create: POST /a2a
Task poll:   GET  /a2a/{task_id}
Task resume: POST /a2a/{task_id}/resume  (for input-required resolution)
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextvars import ContextVar
from typing import Any

import aiosqlite
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route, Router

from recall.db.connection import get_db_path
from recall.worker import _extract_stub_fallback

logger = logging.getLogger(__name__)

# Write-through in-memory cache — DB is the source of truth
_tasks: dict[str, dict[str, Any]] = {}

AGENT_CARD = {
    "name": "recall",
    "version": "0.1.0",
    "description": "Persistent memory layer — consolidates conversation transcripts into typed memories.",
    "defaultInputModes": ["application/json"],
    "defaultOutputModes": ["application/json"],
    "capabilities": {"streaming": False, "pushNotifications": False},
    "skills": [
        {
            "id": "consolidate_memories",
            "name": "Consolidate Memories",
            "description": (
                "Extract and persist memories from conversation transcripts. "
                "Returns input-required when contradictions are detected in existing memories."
            ),
            "inputModes": ["application/json"],
            "outputModes": ["application/json"],
        }
    ],
}

_NEGATION_TOKENS = frozenset(
    {"not", "no", "never", "stopped", "longer", "changed", "removed", "quit", "dropped"}
)


# ── DB helpers ────────────────────────────────────────────────────────────────


async def _db_insert_task(task: dict[str, Any]) -> None:
    async with aiosqlite.connect(get_db_path()) as db:
        await db.execute(
            "INSERT INTO a2a_tasks (id, namespace, status, input, created_at, updated_at) "
            "VALUES (?, ?, 'submitted', ?, datetime('now'), datetime('now'))",
            (task["id"], task["namespace"], json.dumps(task["input"])),
        )
        await db.commit()


async def _db_update_task(task: dict[str, Any]) -> None:
    async with aiosqlite.connect(get_db_path()) as db:
        await db.execute(
            "UPDATE a2a_tasks SET status=?, output=?, message=?, pending=?, resolution=?, "
            "updated_at=datetime('now') WHERE id=?",
            (
                task["status"],
                json.dumps(task["output"]) if task.get("output") is not None else None,
                json.dumps(task["message"]) if task.get("message") is not None else None,
                json.dumps(task["_pending_memories"]) if task.get("_pending_memories") else None,
                task.get("_resolution"),
                task["id"],
            ),
        )
        await db.commit()


async def _db_load_task(task_id: str) -> dict[str, Any] | None:
    async with aiosqlite.connect(get_db_path()) as db:
        rows = await db.execute_fetchall(
            "SELECT id, namespace, status, input, output, message, pending, resolution "
            "FROM a2a_tasks WHERE id = ?",
            (task_id,),
        )
    if not rows:
        return None
    r = rows[0]
    return {
        "id": r[0],
        "namespace": r[1],
        "status": r[2],
        "input": json.loads(r[3]),
        "output": json.loads(r[4]) if r[4] else None,
        "message": json.loads(r[5]) if r[5] else None,
        "_pending_memories": json.loads(r[6]) if r[6] else [],
        "_resolution": r[7],
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────


async def _agent_card(request: Request) -> JSONResponse:
    card = dict(AGENT_CARD)
    # Inject the actual server URL at request time
    base = str(request.base_url).rstrip("/")
    card["url"] = f"{base}/a2a"
    return JSONResponse(card)


async def _create_task(request: Request, namespace_ctx: ContextVar[str]) -> JSONResponse:
    """POST /a2a — submit a consolidation task."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body."}, status_code=400)

    skill = body.get("skill")
    if skill != "consolidate_memories":
        return JSONResponse(
            {"error": f"Unknown skill {skill!r}. Supported: 'consolidate_memories'."},
            status_code=400,
        )

    inp = body.get("input", {})
    text = inp.get("text", "").strip()
    topic = inp.get("topic", "general")
    if not text:
        return JSONResponse({"error": "input.text is required and must not be empty."}, status_code=400)

    task_id = body.get("id") or str(uuid.uuid4())
    namespace = namespace_ctx.get()

    task: dict[str, Any] = {
        "id": task_id,
        "status": "submitted",
        "namespace": namespace,
        "input": {"text": text, "topic": topic},
        "output": None,
        "message": None,
    }
    await _db_insert_task(task)
    _tasks[task_id] = task

    asyncio.create_task(_run_consolidation(task_id), name=f"a2a-consolidate-{task_id[:8]}")
    return JSONResponse({"id": task_id, "status": "submitted"}, status_code=202)


async def _get_task(request: Request) -> JSONResponse:
    """GET /a2a/{task_id} — poll task status."""
    task_id = request.path_params["task_id"]
    task = _tasks.get(task_id)
    if not task:
        task = await _db_load_task(task_id)
        if task:
            _tasks[task_id] = task  # populate cache for future polls
    if not task:
        return JSONResponse({"error": f"Task {task_id!r} not found."}, status_code=404)
    return JSONResponse({
        "id": task["id"],
        "status": task["status"],
        "output": task["output"],
        "message": task.get("message"),
    })


async def _resume_task(request: Request) -> JSONResponse:
    """POST /a2a/{task_id}/resume — resolve input-required contradictions."""
    task_id = request.path_params["task_id"]
    task = _tasks.get(task_id)
    if not task:
        return JSONResponse({"error": f"Task {task_id!r} not found."}, status_code=404)
    if task["status"] != "input-required":
        return JSONResponse(
            {"error": f"Task is in state {task['status']!r}, not 'input-required'."},
            status_code=409,
        )

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body."}, status_code=400)

    resolution = body.get("resolution")
    if resolution not in ("keep_existing", "keep_new", "keep_both"):
        return JSONResponse(
            {"error": "resolution must be 'keep_existing', 'keep_new', or 'keep_both'."},
            status_code=400,
        )

    task["status"] = "working"
    task["_resolution"] = resolution
    await _db_update_task(task)
    asyncio.create_task(_apply_resolution(task_id), name=f"a2a-resolve-{task_id[:8]}")
    return JSONResponse({"id": task_id, "status": "working"})


# ── Background task logic ─────────────────────────────────────────────────────


async def _run_consolidation(task_id: str) -> None:
    task = _tasks.get(task_id)
    if not task:
        return

    task["status"] = "working"
    await _db_update_task(task)
    namespace = task["namespace"]
    text = task["input"]["text"]
    topic = task["input"]["topic"]
    job_id = str(uuid.uuid4())

    # Import lazily to avoid circular import (server.py imports this module)
    import recall.server as _srv
    worker = _srv.extraction_worker

    try:
        if worker is None:
            raise RuntimeError("extraction_worker not started")
        memories = await worker.extract(namespace, text, topic, job_id)
    except Exception as exc:
        logger.warning("a2a_llm_fallback", extra={"task_id": task_id, "error": str(exc)})
        memories = _extract_stub_fallback(namespace, text, topic, job_id)

    if not memories:
        task["status"] = "completed"
        task["output"] = {"memories_stored": 0, "contradictions": []}
        await _db_update_task(task)
        return

    contradictions = await _detect_contradictions(namespace, memories)

    if contradictions:
        task["status"] = "input-required"
        task["_pending_memories"] = memories
        task["message"] = {
            "type": "contradiction_detected",
            "text": (
                f"{len(contradictions)} potential contradiction(s) found. "
                "POST /a2a/{task_id}/resume with resolution: "
                "'keep_existing', 'keep_new', or 'keep_both'."
            ),
            "contradictions": contradictions,
        }
        await _db_update_task(task)
        logger.info(
            "a2a_input_required",
            extra={"task_id": task_id, "contradictions": len(contradictions)},
        )
        return

    stored = await _bulk_store(memories)
    task["status"] = "completed"
    task["output"] = {"memories_stored": stored, "contradictions": []}
    await _db_update_task(task)
    logger.info("a2a_completed", extra={"task_id": task_id, "stored": stored})


async def _apply_resolution(task_id: str) -> None:
    task = _tasks.get(task_id)
    if not task:
        return

    resolution = task.get("_resolution", "keep_both")
    pending: list[dict] = task.get("_pending_memories", [])
    contradictions: list[dict] = (task.get("message") or {}).get("contradictions", [])

    if resolution == "keep_existing":
        conflict_texts = {c["new_text"] for c in contradictions}
        pending = [m for m in pending if m["text"] not in conflict_texts]
    elif resolution == "keep_new":
        conflict_ids = [c["existing_id"] for c in contradictions]
        async with aiosqlite.connect(get_db_path()) as db:
            for mem_id in conflict_ids:
                await db.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
            await db.commit()
    # keep_both: store everything without deleting existing

    stored = await _bulk_store(pending)
    task["status"] = "completed"
    task["output"] = {
        "memories_stored": stored,
        "resolution_applied": resolution,
        "contradictions": contradictions,
    }
    await _db_update_task(task)
    logger.info(
        "a2a_resolved",
        extra={"task_id": task_id, "resolution": resolution, "stored": stored},
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _detect_contradictions(
    namespace: str, new_memories: list[dict]
) -> list[dict]:
    """Heuristic: shared 3+ tokens AND negation word present in either memory."""
    async with aiosqlite.connect(get_db_path()) as db:
        existing = await db.execute_fetchall(
            "SELECT id, text FROM memories WHERE namespace = ?",
            (namespace,),
        )

    contradictions: list[dict] = []
    for new_m in new_memories:
        new_tokens = set(new_m["text"].lower().split())
        for ex_id, ex_text in existing:
            ex_tokens = set(ex_text.lower().split())
            if len(new_tokens & ex_tokens) >= 3:
                if (new_tokens | ex_tokens) & _NEGATION_TOKENS:
                    contradictions.append({
                        "existing_id": ex_id,
                        "existing_text": ex_text,
                        "new_text": new_m["text"],
                    })

    return contradictions


async def _bulk_store(memories: list[dict]) -> int:
    if not memories:
        return 0
    async with aiosqlite.connect(get_db_path()) as db:
        for m in memories:
            await db.execute(
                """INSERT OR IGNORE INTO memories
                   (id, namespace, text, type, topic, importance, confidence,
                    source_session, created_at, entity, attribute, value, valid_from)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    m["id"], m["namespace"], m["text"], m["type"],
                    m["topic"], m["importance"], m["confidence"],
                    m["source_session"], m["created_at"],
                    m.get("entity"), m.get("attribute"), m.get("value"),
                    m.get("valid_from", m["created_at"]),
                ),
            )
        await db.commit()
    return len(memories)


# ── Router factories ──────────────────────────────────────────────────────────


def create_a2a_router(namespace_ctx: ContextVar[str]) -> Router:
    """Create the /a2a task router, bound to the auth context var from server.py."""

    async def create_task(request: Request) -> JSONResponse:
        return await _create_task(request, namespace_ctx)

    return Router(
        routes=[
            Route("/", endpoint=create_task, methods=["POST"]),
            Route("/{task_id}", endpoint=_get_task, methods=["GET"]),
            Route("/{task_id}/resume", endpoint=_resume_task, methods=["POST"]),
        ]
    )


def create_well_known_router() -> Router:
    """Create the /.well-known router for the public Agent Card."""
    return Router(
        routes=[
            Route("/agent-card.json", endpoint=_agent_card, methods=["GET"]),
        ]
    )
