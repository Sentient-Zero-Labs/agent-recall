"""Database backend abstraction for Recall.

SQLiteBackend (default) wraps aiosqlite.
PostgresBackend wraps asyncpg with connection pooling and automatic ? → $N translation.

Usage:
    async with get_backend() as db:
        rows = await db.fetch_all("SELECT id FROM memories WHERE user_id = ?", (uid,))
        await db.execute("UPDATE memories SET ...", (...,))
        await db.commit()

Backend is selected by environment variable:
  RECALL_DB_URL set  → PostgresBackend (asyncpg)
  RECALL_DB_URL unset → SQLiteBackend  (aiosqlite, default)
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from typing import Any


class DatabaseBackend(ABC):
    @abstractmethod
    async def execute(self, sql: str, params: tuple = ()) -> int:
        """Execute a statement and return rowcount."""

    @abstractmethod
    async def fetch_all(self, sql: str, params: tuple = ()) -> list[tuple]:
        """Fetch all rows matching the query."""

    @abstractmethod
    async def fetch_one(self, sql: str, params: tuple = ()) -> tuple | None:
        """Fetch a single row or None."""

    @abstractmethod
    async def executemany(self, sql: str, params_list: list[tuple]) -> None:
        """Execute a statement once per params tuple."""

    @abstractmethod
    async def commit(self) -> None:
        """Commit the current transaction."""

    @abstractmethod
    async def __aenter__(self) -> "DatabaseBackend": ...

    @abstractmethod
    async def __aexit__(self, *args: Any) -> None: ...


# ── SQLite ────────────────────────────────────────────────────────────────────


class SQLiteBackend(DatabaseBackend):
    """Thin aiosqlite wrapper. Uses ? placeholders (SQLite default)."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn: Any = None  # aiosqlite.Connection

    async def __aenter__(self) -> "SQLiteBackend":
        import aiosqlite
        self._conn = await aiosqlite.connect(self._db_path)
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def execute(self, sql: str, params: tuple = ()) -> int:
        cursor = await self._conn.execute(sql, params)
        return cursor.rowcount

    async def fetch_all(self, sql: str, params: tuple = ()) -> list[tuple]:
        return await self._conn.execute_fetchall(sql, params)

    async def fetch_one(self, sql: str, params: tuple = ()) -> tuple | None:
        rows = await self._conn.execute_fetchall(sql, params)
        return rows[0] if rows else None

    async def executemany(self, sql: str, params_list: list[tuple]) -> None:
        await self._conn.executemany(sql, params_list)

    async def commit(self) -> None:
        await self._conn.commit()


# ── Postgres ──────────────────────────────────────────────────────────────────

_PLACEHOLDER_RE = re.compile(r"\?")


def _translate_placeholders(sql: str) -> str:
    """Replace SQLite ? placeholders with Postgres $1, $2, ... positional params."""
    counter = 0

    def _replace(_match: re.Match) -> str:
        nonlocal counter
        counter += 1
        return f"${counter}"

    return _PLACEHOLDER_RE.sub(_replace, sql)


class PostgresBackend(DatabaseBackend):
    """asyncpg wrapper with connection pooling and automatic ? → $N translation."""

    def __init__(self, url: str) -> None:
        self._url = url
        self._pool: Any = None  # asyncpg.Pool
        self._conn: Any = None  # asyncpg.Connection acquired from pool

    async def _ensure_pool(self) -> None:
        if self._pool is None:
            import asyncpg
            self._pool = await asyncpg.create_pool(self._url, min_size=1, max_size=10)

    async def __aenter__(self) -> "PostgresBackend":
        await self._ensure_pool()
        self._conn = await self._pool.acquire()
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._conn is not None:
            await self._pool.release(self._conn)
            self._conn = None

    async def execute(self, sql: str, params: tuple = ()) -> int:
        result = await self._conn.execute(_translate_placeholders(sql), *params)
        # asyncpg returns e.g. "UPDATE 3" — extract the count
        try:
            return int(result.split()[-1])
        except (ValueError, IndexError):
            return 0

    async def fetch_all(self, sql: str, params: tuple = ()) -> list[tuple]:
        rows = await self._conn.fetch(_translate_placeholders(sql), *params)
        return [tuple(r) for r in rows]

    async def fetch_one(self, sql: str, params: tuple = ()) -> tuple | None:
        row = await self._conn.fetchrow(_translate_placeholders(sql), *params)
        return tuple(row) if row else None

    async def executemany(self, sql: str, params_list: list[tuple]) -> None:
        await self._conn.executemany(_translate_placeholders(sql), params_list)

    async def commit(self) -> None:
        pass  # asyncpg uses implicit transactions; explicit commit via transaction context


# ── Factory ───────────────────────────────────────────────────────────────────


def get_backend() -> DatabaseBackend:
    """Return the appropriate backend based on RECALL_DB_URL environment variable.

    RECALL_DB_URL set  → PostgresBackend
    RECALL_DB_URL unset → SQLiteBackend (uses RECALL_DB_PATH / recall.db)
    """
    url = os.environ.get("RECALL_DB_URL")
    if url:
        return PostgresBackend(url)
    from recall.db.connection import get_db_path
    return SQLiteBackend(str(get_db_path()))
