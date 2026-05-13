"""Recall extraction worker — processes store_memory jobs from the async queue."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import aiosqlite
import anthropic

from recall.db.connection import get_db_path

logger = logging.getLogger(__name__)

_WORKER_QUEUE_MAX = 1000

_EXTRACTION_PROMPT = """\
Extract important memories from this conversation text. Return ONLY a JSON array.

Each memory object must have:
- "text": the memory content (1-2 sentences, specific and self-contained)
- "type": one of "preference", "fact", "decision", "procedure"
- "importance": float 0.0-1.0 (how important is this to remember long-term?)
- "confidence": float 0.0-1.0 (how certain are you this is accurate from the text?)
- "entity": the subject of the fact — e.g. "user", "project-alpha", "tool-x" (null if not applicable)
- "attribute": the property — e.g. "preferred_language", "works_at", "uses_framework" (null if not applicable)
- "value": the value — e.g. "Python", "Acme Corp", "FastAPI" (null if not applicable)

Rules:
- For preferences and facts: always try to extract entity/attribute/value.
- For decisions and procedures: entity/attribute/value are usually null.
- If entity+attribute+value are set, they must be consistent with the text field.
- Ignore small talk, pleasantries, and purely transient information.
- Extract 0-5 memories. Return [] if nothing is worth remembering.

Topic context: {topic}

Text:
{text}"""

_VALID_TYPES = {"preference", "fact", "decision", "procedure"}


class ExtractionWorker:
    """Processes extraction jobs dequeued from an in-memory asyncio.Queue.

    Uses the Anthropic API (claude-haiku-4-5-20251001) for LLM-powered extraction.
    Falls back to _extract_stub_fallback on any API or parse failure.
    """

    def __init__(self) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Set it before starting the Recall server."
            )
        self._api_key = api_key
        self._client: anthropic.AsyncAnthropic | None = None
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_WORKER_QUEUE_MAX)
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
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
        if self._client:
            await self._client.close()
        logger.info("extraction_worker_stopped")

    async def extract(
        self, namespace: str, text: str, topic: str, job_id: str
    ) -> list[dict]:
        """Public extraction interface — used by A2A and other synchronous callers."""
        return await self._extract_with_llm(namespace, text, topic, job_id)

    async def enqueue(self, job: dict[str, Any]) -> None:
        """Add a job to the extraction queue. Non-blocking: raises QueueFull if at capacity."""
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull:
            logger.warning(
                "extraction_queue_full",
                extra={"job_id": job.get("job_id"), "namespace": job.get("namespace")},
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
        """Process one extraction job using LLM extraction with stub fallback."""
        job_id = job["job_id"]
        namespace = job["namespace"]
        text = job["text"]
        topic = job.get("topic", "general")
        session_id = job.get("session_id", "")
        agent_id = job.get("agent_id", "")

        async with aiosqlite.connect(get_db_path()) as db:
            await db.execute(
                "UPDATE operations SET status = 'processing', updated_at = datetime('now') "
                "WHERE id = ?",
                (job_id,),
            )
            await db.commit()

        try:
            memories = await self._extract_with_llm(namespace, text, topic, job_id)
        except Exception as exc:
            logger.error(
                "extraction_llm_failed",
                extra={"job_id": job_id, "namespace": namespace, "error": str(exc)},
                exc_info=True,
            )
            memories = _extract_stub_fallback(namespace, text, topic, job_id)

        async with aiosqlite.connect(get_db_path()) as db:
            for memory in memories:
                await _handle_contradiction(db, memory, namespace)
                await db.execute(
                    """INSERT OR IGNORE INTO memories
                       (id, namespace, text, type, topic, importance, confidence,
                        source_session, created_at, entity, attribute, value,
                        valid_from, session_id, agent_id)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        memory["id"],
                        memory["namespace"],
                        memory["text"],
                        memory["type"],
                        memory["topic"],
                        memory["importance"],
                        memory["confidence"],
                        memory["source_session"],
                        memory["created_at"],
                        memory.get("entity"),
                        memory.get("attribute"),
                        memory.get("value"),
                        memory.get("valid_from"),
                        session_id or None,
                        agent_id or None,
                    ),
                )
            await db.execute(
                "UPDATE operations SET status = 'complete', updated_at = datetime('now') "
                "WHERE id = ?",
                (job_id,),
            )
            await db.commit()

        logger.info(
            "extraction_complete",
            extra={"job_id": job_id, "namespace": namespace, "memories_stored": len(memories)},
        )

    async def _extract_with_llm(
        self, namespace: str, text: str, topic: str, source_session: str
    ) -> list[dict]:
        """Call the Anthropic API to extract structured memories from conversation text.

        Returns a list of memory dicts ready for DB insertion.
        Raises on API failure or JSON parse error — caller must handle and fall back.
        """
        assert self._client is not None, "start() must be called before extraction"

        prompt = _EXTRACTION_PROMPT.format(topic=topic, text=text)

        response = await self._client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        raw_content = response.content[0].text.strip()
        # Strip markdown code fences the model sometimes adds despite the prompt
        if raw_content.startswith("```"):
            raw_content = raw_content.split("\n", 1)[-1]
            raw_content = raw_content.rsplit("```", 1)[0].strip()

        try:
            extracted = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM returned non-JSON response: {raw_content[:200]!r}"
            ) from exc

        if not isinstance(extracted, list):
            raise ValueError(
                f"LLM returned non-list JSON (got {type(extracted).__name__}): "
                f"{raw_content[:200]!r}"
            )

        now = datetime.now(timezone.utc).isoformat()
        memories: list[dict] = []

        for item in extracted:
            if not isinstance(item, dict):
                logger.warning(
                    "extraction_llm_invalid_item",
                    extra={"job_id": source_session, "item": str(item)[:100]},
                )
                continue

            mem_type = item.get("type", "fact")
            if mem_type not in _VALID_TYPES:
                logger.warning(
                    "extraction_llm_unknown_type",
                    extra={"job_id": source_session, "type": mem_type},
                )
                mem_type = "fact"

            importance = float(item.get("importance", 0.5))
            confidence = float(item.get("confidence", 0.8))
            importance = max(0.0, min(1.0, importance))
            confidence = max(0.0, min(1.0, confidence))

            mem_text = str(item.get("text", "")).strip()
            if not mem_text:
                continue

            memories.append(
                {
                    "id": str(uuid.uuid4()),
                    "namespace": namespace,
                    "text": mem_text,
                    "type": mem_type,
                    "topic": topic,
                    "importance": importance,
                    "confidence": confidence,
                    "source_session": source_session,
                    "created_at": now,
                    "entity": item.get("entity") or None,
                    "attribute": item.get("attribute") or None,
                    "value": item.get("value") or None,
                    "valid_from": now,
                }
            )

        logger.info(
            "extraction_llm_success",
            extra={
                "job_id": source_session,
                "namespace": namespace,
                "memories_extracted": len(memories),
            },
        )
        return memories


async def _handle_contradiction(db: Any, memory: dict, namespace: str) -> None:
    """Mark conflicting active memories as superseded when entity+attribute match but value differs."""
    if not memory.get("entity") or not memory.get("attribute"):
        return
    existing = await db.execute_fetchall(
        "SELECT id FROM memories WHERE namespace = ? AND entity = ? AND attribute = ? "
        "AND valid_until IS NULL AND (value IS NULL OR value != ?)",
        (namespace, memory["entity"], memory["attribute"], memory["value"]),
    )
    for (ex_id,) in existing:
        await db.execute(
            "UPDATE memories SET valid_until = ?, superseded_by = ? WHERE id = ?",
            (memory["created_at"], memory["id"], ex_id),
        )


def _extract_stub_fallback(
    namespace: str, text: str, topic: str, source_session: str
) -> list[dict]:
    """Fallback stub — returns the raw text as a single fact memory when LLM extraction fails."""
    now = datetime.now(timezone.utc).isoformat()
    return [
        {
            "id": str(uuid.uuid4()),
            "namespace": namespace,
            "text": text[:500],  # truncate to 500 chars
            "type": "fact",
            "topic": topic,
            "importance": 0.5,
            "confidence": 0.8,
            "source_session": source_session,
            "created_at": now,
            "entity": None,
            "attribute": None,
            "value": None,
            "valid_from": now,
        }
    ]
