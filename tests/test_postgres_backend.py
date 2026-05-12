"""Database backend abstraction tests.

Tests SQLiteBackend CRUD, placeholder translation, and factory selection.
PostgresBackend live-connection tests are skipped unless RECALL_DB_URL is set.
"""

from __future__ import annotations

import os
import uuid

import pytest

from recall.db.backend import (
    PostgresBackend,
    SQLiteBackend,
    _translate_placeholders,
    get_backend,
)


class TestPlaceholderTranslation:
    def test_single_placeholder(self):
        assert _translate_placeholders("SELECT * FROM t WHERE id = ?") == \
               "SELECT * FROM t WHERE id = $1"

    def test_multiple_placeholders(self):
        sql = "INSERT INTO t (a, b, c) VALUES (?, ?, ?)"
        assert _translate_placeholders(sql) == "INSERT INTO t (a, b, c) VALUES ($1, $2, $3)"

    def test_no_placeholders_unchanged(self):
        sql = "SELECT * FROM t"
        assert _translate_placeholders(sql) == sql

    def test_ten_placeholders(self):
        sql = "?,?,?,?,?,?,?,?,?,?"
        result = _translate_placeholders(sql)
        assert result == "$1,$2,$3,$4,$5,$6,$7,$8,$9,$10"


class TestSQLiteBackend:
    async def test_execute_and_fetch_all(self, tmp_db):
        """SQLiteBackend can create a table, insert, and fetch rows."""
        async with SQLiteBackend(str(tmp_db)) as db:
            await db.execute(
                "CREATE TABLE IF NOT EXISTS _test (id TEXT PRIMARY KEY, val TEXT)"
            )
            uid = str(uuid.uuid4())
            await db.execute("INSERT INTO _test (id, val) VALUES (?, ?)", (uid, "hello"))
            await db.commit()
            rows = await db.fetch_all("SELECT id, val FROM _test WHERE id = ?", (uid,))

        assert len(rows) == 1
        assert rows[0][0] == uid
        assert rows[0][1] == "hello"

    async def test_fetch_one_returns_single_row(self, tmp_db):
        async with SQLiteBackend(str(tmp_db)) as db:
            await db.execute("CREATE TABLE IF NOT EXISTS _t2 (id TEXT PRIMARY KEY)")
            uid = str(uuid.uuid4())
            await db.execute("INSERT INTO _t2 (id) VALUES (?)", (uid,))
            await db.commit()
            row = await db.fetch_one("SELECT id FROM _t2 WHERE id = ?", (uid,))
            none = await db.fetch_one("SELECT id FROM _t2 WHERE id = ?", ("nope",))

        assert row is not None
        assert row[0] == uid
        assert none is None

    async def test_execute_returns_rowcount(self, tmp_db):
        async with SQLiteBackend(str(tmp_db)) as db:
            await db.execute("CREATE TABLE IF NOT EXISTS _t3 (id TEXT PRIMARY KEY)")
            await db.execute("INSERT INTO _t3 (id) VALUES (?)", (str(uuid.uuid4()),))
            await db.execute("INSERT INTO _t3 (id) VALUES (?)", (str(uuid.uuid4()),))
            await db.commit()
            count = await db.execute("DELETE FROM _t3")
            await db.commit()

        assert count == 2


class TestBackendFactory:
    def test_returns_sqlite_by_default(self, monkeypatch):
        """Without RECALL_DB_URL, factory returns SQLiteBackend."""
        monkeypatch.delenv("RECALL_DB_URL", raising=False)
        backend = get_backend()
        assert isinstance(backend, SQLiteBackend)

    def test_returns_postgres_when_url_set(self, monkeypatch):
        """With RECALL_DB_URL set, factory returns PostgresBackend (no connection yet)."""
        monkeypatch.setenv("RECALL_DB_URL", "postgresql://user:pass@localhost/testdb")
        backend = get_backend()
        assert isinstance(backend, PostgresBackend)
        # Pool is not created until __aenter__ — just check the URL was stored
        assert "localhost" in backend._url


@pytest.mark.skipif(
    not os.environ.get("RECALL_DB_URL"),
    reason="RECALL_DB_URL not set — skip live Postgres tests",
)
class TestPostgresBackendLive:
    async def test_postgres_connects_and_queries(self):
        """Connects to a real Postgres instance and runs a basic query."""
        async with PostgresBackend(os.environ["RECALL_DB_URL"]) as db:
            row = await db.fetch_one("SELECT 1 AS val")
        assert row is not None
        assert row[0] == 1
