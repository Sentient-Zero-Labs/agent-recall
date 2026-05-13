"""Extraction worker tests — real Anthropic API calls where needed.

Tests are skipped automatically when ANTHROPIC_API_KEY is not set,
so they don't break CI without credentials.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from recall.worker import ExtractionWorker, _extract_stub_fallback

_needs_api = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skip live extraction tests",
)


@_needs_api
class TestExtraction:
    async def test_extract_returns_typed_memories(self, client, namespace):
        """LLM extraction returns structurally valid memories — asserts schema, not content."""
        worker = ExtractionWorker()
        await worker.start()
        try:
            memories = await worker.extract(
                namespace=namespace,
                text=(
                    "I prefer Python over Go for backend services. "
                    "I always write tests before shipping."
                ),
                topic="engineering",
                job_id="test-job-001",
            )
        finally:
            await worker.stop()

        assert isinstance(memories, list)
        assert len(memories) >= 1
        for m in memories:
            assert m["type"] in ("preference", "fact", "decision", "procedure")
            assert 0.0 <= m["importance"] <= 1.0
            assert 0.0 <= m["confidence"] <= 1.0
            assert len(m["text"]) > 0

    async def test_full_pipeline_stores_memories_to_db(self, client, namespace, tmp_db):
        """Queue → worker → DB path: enqueue a job, poll until memories persist (15s max)."""
        from recall.db.connection import set_db_path
        set_db_path(tmp_db)

        worker = ExtractionWorker()
        await worker.start()
        job_id = str(uuid.uuid4())
        await worker.enqueue({
            "job_id": job_id,
            "namespace": namespace,
            "text": (
                "I always use type hints in Python. "
                "FastAPI is my preferred framework for new services."
            ),
            "topic": "programming",
        })

        memories = []
        for _ in range(50):
            await asyncio.sleep(0.3)
            memories, _ = await client.list_all(namespace)
            if memories:
                break

        await worker.stop()
        assert len(memories) > 0, "Pipeline should have extracted and persisted at least one memory"


class TestStubFallback:
    async def test_stub_fallback_returns_raw_text_as_fact(self):
        """Fallback stores raw text as a fact — no API call."""
        memories = _extract_stub_fallback("u1", "Some raw conversation text", "general", "job-fallback")
        assert len(memories) == 1
        assert memories[0]["type"] == "fact"
        assert memories[0]["text"] == "Some raw conversation text"
        assert memories[0]["namespace"] == "u1"

    async def test_stub_fallback_truncates_long_text(self):
        long_text = "x" * 1000
        memories = _extract_stub_fallback("u1", long_text, "general", "job-trunc")
        assert len(memories[0]["text"]) == 500
