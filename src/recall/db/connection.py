"""SQLite connection management for Recall."""

from __future__ import annotations

import os
from pathlib import Path

import aiosqlite

_DB_PATH: Path = Path(os.environ.get("RECALL_DB_PATH", "recall.db"))
_SCHEMA_PATH: Path = Path(__file__).parent / "schema.sql"

# Columns added in schema v2 — applied via ALTER TABLE for existing DBs
_V2_COLUMNS = [
    ("entity", "TEXT"),
    ("attribute", "TEXT"),
    ("value", "TEXT"),
    ("valid_from", "TEXT"),
    ("valid_until", "TEXT"),
    ("session_id", "TEXT"),
    ("agent_id", "TEXT"),
    ("linked_ids", "TEXT"),
]


def set_db_path(path: str | Path) -> None:
    global _DB_PATH
    _DB_PATH = Path(path)


async def _migrate_v2(db: aiosqlite.Connection) -> None:
    """Add v2 structured fact columns to an existing memories table."""
    for col, coltype in _V2_COLUMNS:
        try:
            await db.execute(f"ALTER TABLE memories ADD COLUMN {col} {coltype}")
        except Exception:
            pass  # column already exists — safe to ignore
    try:
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_entity_attr "
            "ON memories(user_id, entity, attribute) WHERE valid_until IS NULL"
        )
    except Exception:
        pass
    await db.commit()


async def init_db() -> None:
    """Create tables and apply schema. Idempotent — safe to call on every startup."""
    schema = _SCHEMA_PATH.read_text()
    async with aiosqlite.connect(_DB_PATH) as db:
        await db.executescript(schema)

        # WAL mode for concurrent reads + writes without blocking
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA synchronous=NORMAL")
        await db.execute("PRAGMA foreign_keys=ON")
        await db.commit()

        await _migrate_v2(db)


def get_db_path() -> Path:
    return _DB_PATH
