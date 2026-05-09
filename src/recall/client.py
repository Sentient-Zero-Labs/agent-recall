"""MemoryClient — Python client for direct Recall database access.

Used in tests, scripts, and agent frameworks that embed Recall locally
without an HTTP server. For MCP-based access, use the FastMCP server.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from datetime import datetime

import aiosqlite

from recall.db.connection import get_db_path, init_db
from recall.models import MemoryType, MemoryUnit


class MemoryClient:
    """Async client for direct SQLite operations on the Recall memory store.

    Typical use: tests, admin scripts, local agent integration.
    """

    def __init__(self, db_path: str | None = None) -> None:
        if db_path:
            from recall.db.connection import set_db_path
            set_db_path(db_path)

    async def initialize(self) -> None:
        """Create tables. Call once before using any other method."""
        await init_db()

    # ── Write ──────────────────────────────────────────────────────────────

    async def store(
        self,
        user_id: str,
        text: str,
        topic: str,
        memory_type: MemoryType = MemoryType.FACT,
        importance: float = 0.5,
        confidence: float = 0.8,
        source_session: str = "",
    ) -> MemoryUnit:
        """Store a memory directly (synchronous, no extraction worker).
        For production use store_memory MCP tool with async extraction.
        """
        memory = MemoryUnit(
            id=str(uuid.uuid4()),
            user_id=user_id,
            text=text,
            type=memory_type,
            topic=topic,
            importance=importance,
            confidence=confidence,
            source_session=source_session,
            created_at=datetime.utcnow(),
        )
        async with aiosqlite.connect(get_db_path()) as db:
            await db.execute(
                """INSERT INTO memories
                   (id, user_id, text, type, topic, importance, confidence,
                    source_session, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    memory.id,
                    memory.user_id,
                    memory.text,
                    memory.type.value,
                    memory.topic,
                    memory.importance,
                    memory.confidence,
                    memory.source_session,
                    memory.created_at.isoformat(),
                ),
            )
            await db.commit()
        return memory

    # ── Read ───────────────────────────────────────────────────────────────

    async def get(self, memory_id: str, user_id: str) -> MemoryUnit | None:
        async with aiosqlite.connect(get_db_path()) as db:
            rows = await db.execute_fetchall(
                "SELECT id, user_id, text, type, topic, importance, confidence, "
                "source_session, created_at, last_accessed, access_count "
                "FROM memories WHERE id = ? AND user_id = ?",
                (memory_id, user_id),
            )
        if not rows:
            return None
        return _row_to_memory_unit(rows[0])

    async def search(
        self, user_id: str, query: str, limit: int = 20
    ) -> list[MemoryUnit]:
        """Simple keyword search. Thin wrapper — use MCP tool for production."""
        async with aiosqlite.connect(get_db_path()) as db:
            rows = await db.execute_fetchall(
                "SELECT id, user_id, text, type, topic, importance, confidence, "
                "source_session, created_at, last_accessed, access_count "
                "FROM memories WHERE user_id = ? AND text LIKE ? LIMIT ?",
                (user_id, f"%{query}%", limit),
            )
        return [_row_to_memory_unit(r) for r in rows]

    async def list_all(
        self, user_id: str, limit: int = 20, offset: int = 0
    ) -> tuple[list[MemoryUnit], int]:
        """Return (memories, total_count)."""
        async with aiosqlite.connect(get_db_path()) as db:
            rows = await db.execute_fetchall(
                "SELECT id, user_id, text, type, topic, importance, confidence, "
                "source_session, created_at, last_accessed, access_count "
                "FROM memories WHERE user_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (user_id, limit, offset),
            )
            count_rows = await db.execute_fetchall(
                "SELECT COUNT(*) FROM memories WHERE user_id = ?", (user_id,)
            )
        total = count_rows[0][0]
        return [_row_to_memory_unit(r) for r in rows], total

    # ── Delete ─────────────────────────────────────────────────────────────

    async def delete(self, memory_id: str, user_id: str) -> bool:
        """Returns True if deleted, False if not found."""
        async with aiosqlite.connect(get_db_path()) as db:
            cursor = await db.execute(
                "DELETE FROM memories WHERE id = ? AND user_id = ?",
                (memory_id, user_id),
            )
            await db.commit()
        return cursor.rowcount > 0

    async def delete_all(self, user_id: str) -> int:
        """Delete all memories for a user. Returns count deleted."""
        async with aiosqlite.connect(get_db_path()) as db:
            cursor = await db.execute(
                "DELETE FROM memories WHERE user_id = ?", (user_id,)
            )
            await db.commit()
        return cursor.rowcount


def _row_to_memory_unit(row: tuple) -> MemoryUnit:
    return MemoryUnit(
        id=row[0],
        user_id=row[1],
        text=row[2],
        type=MemoryType(row[3]),
        topic=row[4] or "",
        importance=row[5],
        confidence=row[6],
        source_session=row[7] or "",
        created_at=datetime.fromisoformat(row[8]),
        last_accessed=datetime.fromisoformat(row[9]) if row[9] else None,
        access_count=row[10] or 0,
    )
