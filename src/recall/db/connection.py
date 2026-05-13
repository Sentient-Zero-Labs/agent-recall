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
            "ON memories(namespace, entity, attribute) WHERE valid_until IS NULL"
        )
    except Exception:
        pass
    await db.commit()


async def _migrate_v3(db: aiosqlite.Connection) -> None:
    """Create a2a_tasks table for persistent A2A task storage (v3)."""
    await db.execute(
        """CREATE TABLE IF NOT EXISTS a2a_tasks (
            id          TEXT PRIMARY KEY,
            namespace     TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'submitted',
            input       TEXT NOT NULL,
            output      TEXT,
            message     TEXT,
            pending     TEXT,
            resolution  TEXT,
            created_at  TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
        )"""
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_a2a_user ON a2a_tasks(namespace)"
    )
    await db.commit()


async def _migrate_v4(db: aiosqlite.Connection) -> None:
    """Rename user_id → namespace across all tables (v4 — SQLite 3.25+ RENAME COLUMN)."""
    renames = [
        ("memories",          "user_id", "namespace"),
        ("operations",        "user_id", "namespace"),
        ("api_tokens",        "user_id", "namespace"),
        ("tool_call_records", "user_id", "namespace"),
        ("a2a_tasks",         "user_id", "namespace"),
    ]
    for table, old_col, new_col in renames:
        try:
            # Check if old column still exists before renaming
            info = await db.execute_fetchall(f"PRAGMA table_info({table})")
            cols = [row[1] for row in info]
            if old_col in cols and new_col not in cols:
                await db.execute(
                    f"ALTER TABLE {table} RENAME COLUMN {old_col} TO {new_col}"
                )
        except Exception:
            pass  # already renamed or table doesn't exist
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
        await _migrate_v3(db)
        await _migrate_v4(db)


def get_db_path() -> Path:
    return _DB_PATH
