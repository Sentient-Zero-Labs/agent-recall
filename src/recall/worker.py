"""Recall extraction worker — processes store_memory jobs from the async queue."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any

import aiosqlite

from recall.db.connection import get_db_path

logger = logging.getLogger(__name__)

_WORKER_QUEUE_MAX = 1000


class ExtractionWorker:
    """Processes extraction jobs dequeued from an in-memory asyncio.Queue.

    v0.1: LLM extraction is stubbed — memories are stored with placeholder values.
    Full extraction pipeline (Claude API + structured output) is the v0.2 upgrade
    documented in tools-series-scope.md.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_WORKER_QUEUE_MAX)
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run(), name="recall-extraction-worker")
        logger.info("extraction_worker_started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("extraction_worker_stopped")

    async def enqueue(self, job: dict[str, Any]) -> None:
        """Add a job to the extraction queue. Non-blocking: raises QueueFull if at capacity."""
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull:
            logger.warning(
                "extraction_queue_full",
                extra={"job_id": job.get("job_id"), "user_id": job.get("user_id")},
            )
            raise

    async def _run(self) -> None:
        while self._running:
            try:
                job = await self._queue.get()
                try:
                    await self._process(job)
                except Exception as exc:
                    logger.error(
                        "extraction_job_failed",
                        extra={"job_id": job.get("job_id"), "error": str(exc)},
                        exc_info=True,
                    )
                finally:
                    self._queue.task_done()
            except asyncio.CancelledError:
                break

    async def _process(self, job: dict[str, Any]) -> None:
        """Process one extraction job. Stub in v0.1 — stores the text as a single memory."""
        job_id = job["job_id"]
        user_id = job["user_id"]
        text = job["text"]
        topic = job.get("topic", "general")

        async with aiosqlite.connect(get_db_path()) as db:
            await db.execute(
                "UPDATE operations SET status = 'processing' WHERE id = ?",
                (job_id,),
            )
            await db.commit()

        # v0.1 stub: store the raw text as a single FACT memory.
        # v0.2: call Claude API for structured extraction → multiple MemoryUnits.
        memories = _extract_stub(user_id, text, topic, job_id)

        async with aiosqlite.connect(get_db_path()) as db:
            for memory in memories:
                await db.execute(
                    """INSERT OR IGNORE INTO memories
                       (id, user_id, text, type, topic, importance, confidence,
                        source_session, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        memory["id"],
                        memory["user_id"],
                        memory["text"],
                        memory["type"],
                        memory["topic"],
                        memory["importance"],
                        memory["confidence"],
                        memory["source_session"],
                        memory["created_at"],
                    ),
                )
            await db.execute(
                "UPDATE operations SET status = 'complete' WHERE id = ?",
                (job_id,),
            )
            await db.commit()

        logger.info(
            "extraction_complete",
            extra={"job_id": job_id, "user_id": user_id, "memories_stored": len(memories)},
        )


def _extract_stub(
    user_id: str, text: str, topic: str, source_session: str
) -> list[dict]:
    """v0.1 stub — returns the raw text as a single fact memory."""
    return [
        {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "text": text[:500],  # truncate to 500 chars for v0.1
            "type": "fact",
            "topic": topic,
            "importance": 0.5,
            "confidence": 0.8,
            "source_session": source_session,
            "created_at": datetime.utcnow().isoformat(),
        }
    ]
