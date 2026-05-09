-- Recall SQLite schema
-- Version: 1 (Tools series baseline)
-- Apply with: aiosqlite execute_script() on first startup
-- Postgres migration: Memory series Issue 01

-- ------------------------------------------------------------------ --
-- Schema version tracking                                             --
-- ------------------------------------------------------------------ --

CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  TEXT NOT NULL,
    description TEXT NOT NULL
);

INSERT OR IGNORE INTO schema_version (version, applied_at, description)
VALUES (1, datetime('now'), 'Initial schema — memories + tool_call_records');

-- ------------------------------------------------------------------ --
-- Core memory store                                                   --
-- ------------------------------------------------------------------ --

CREATE TABLE IF NOT EXISTS memories (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    text            TEXT NOT NULL,
    type            TEXT NOT NULL,          -- preference|fact|decision|procedure
    topic           TEXT,
    importance      REAL DEFAULT 0.5,
    confidence      REAL DEFAULT 0.8,
    source_session  TEXT,
    created_at      TEXT NOT NULL,
    last_accessed   TEXT,
    access_count    INTEGER DEFAULT 0,
    decay_score     REAL,                   -- computed in Memory series
    superseded_by   TEXT,                   -- contradiction resolution in Memory series
    embedding       BLOB                    -- added when pgvector available
);

-- ------------------------------------------------------------------ --
-- Observability: tool call audit log                                  --
-- ------------------------------------------------------------------ --

CREATE TABLE IF NOT EXISTS tool_call_records (
    id              TEXT PRIMARY KEY,
    tool_name       TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    inputs_hash     TEXT NOT NULL,          -- sha256 of canonicalised inputs
    status          TEXT NOT NULL,          -- success|error|timeout
    error_code      TEXT,
    duration_ms     INTEGER NOT NULL,
    llm_tokens_in   INTEGER DEFAULT 0,
    llm_tokens_out  INTEGER DEFAULT 0,
    cost_usd        REAL DEFAULT 0.0,
    timestamp       TEXT NOT NULL
);

-- ------------------------------------------------------------------ --
-- Indexes                                                              --
-- ------------------------------------------------------------------ --

CREATE INDEX IF NOT EXISTS idx_memories_user
    ON memories(user_id);

CREATE INDEX IF NOT EXISTS idx_memories_user_type
    ON memories(user_id, type);

CREATE INDEX IF NOT EXISTS idx_memories_created
    ON memories(created_at);

CREATE INDEX IF NOT EXISTS idx_tool_calls_user_session
    ON tool_call_records(user_id, session_id);

CREATE INDEX IF NOT EXISTS idx_tool_calls_timestamp
    ON tool_call_records(timestamp);

CREATE INDEX IF NOT EXISTS idx_tool_calls_tool_name
    ON tool_call_records(tool_name);

-- ------------------------------------------------------------------ --
-- Idempotency + extraction job tracking                               --
-- ------------------------------------------------------------------ --

CREATE TABLE IF NOT EXISTS operations (
    id              TEXT PRIMARY KEY,
    idempotency_key TEXT UNIQUE NOT NULL,
    user_id         TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'queued', -- queued|processing|complete|failed
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_operations_user
    ON operations(user_id);

CREATE INDEX IF NOT EXISTS idx_operations_status
    ON operations(status);

-- ------------------------------------------------------------------ --
-- API token store                                                     --
-- ------------------------------------------------------------------ --

CREATE TABLE IF NOT EXISTS api_tokens (
    id          TEXT PRIMARY KEY,
    token_hash  TEXT UNIQUE NOT NULL,   -- sha256 of the raw token
    user_id     TEXT NOT NULL,
    revoked     INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_api_tokens_hash
    ON api_tokens(token_hash);
