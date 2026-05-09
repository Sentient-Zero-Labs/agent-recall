"""SQLite connection management for Recall."""

from __future__ import annotations

import os
from pathlib import Path

import aiosqlite

_DB_PATH: Path = Path(os.environ.get("RECALL_DB_PATH", "recall.db"))
_SCHEMA_PATH: Path = Path(__file__).parent / "schema.sql"


def set_db_path(path: str | Path) -> None:
    global _DB_PATH
    _DB_PATH = Path(path)


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


def get_db_path() -> Path:
    return _DB_PATH
