"""ToolCallRecord dataclass and LoggingMiddleware for Recall."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import aiosqlite
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from recall.db.connection import get_db_path

session_id_ctx: ContextVar[str] = ContextVar("session_id", default="")
token_in_ctx: ContextVar[int] = ContextVar("token_in", default=0)
token_out_ctx: ContextVar[int] = ContextVar("token_out", default=0)

_MODEL_RATES: dict[str, tuple[float, float]] = {
    "claude-sonnet": (3.00 / 1_000_000, 15.00 / 1_000_000),
    "claude-haiku": (0.25 / 1_000_000, 1.25 / 1_000_000),
    "claude-opus": (15.00 / 1_000_000, 75.00 / 1_000_000),
}


@dataclass
class ToolCallRecord:
    id: str
    tool_name: str
    namespace: str
    session_id: str
    inputs_hash: str
    status: Literal["success", "error", "timeout"]
    error_code: str | None
    duration_ms: int
    llm_tokens_in: int
    llm_tokens_out: int
    cost_usd: float
    timestamp: datetime


def estimate_cost(tokens_in: int, tokens_out: int, model: str = "claude-sonnet") -> float:
    """Estimate cost from token counts within ~5% of actual billing."""
    rate_in, rate_out = _MODEL_RATES.get(model, _MODEL_RATES["claude-sonnet"])
    return round(tokens_in * rate_in + tokens_out * rate_out, 6)


def hash_inputs(inputs: Any) -> str:
    """16-char prefix of sha256 — detects duplicates without storing content."""
    serialized = json.dumps(inputs, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


async def insert_tool_call_record(record: ToolCallRecord) -> None:
    async with aiosqlite.connect(get_db_path()) as db:
        await db.execute(
            """INSERT INTO tool_call_records
               (id, tool_name, namespace, session_id, inputs_hash, status, error_code,
                duration_ms, llm_tokens_in, llm_tokens_out, cost_usd, timestamp)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                record.id,
                record.tool_name,
                record.namespace,
                record.session_id,
                record.inputs_hash,
                record.status,
                record.error_code,
                record.duration_ms,
                record.llm_tokens_in,
                record.llm_tokens_out,
                record.cost_usd,
                record.timestamp.isoformat(),
            ),
        )
        await db.commit()


class LoggingMiddleware(BaseHTTPMiddleware):
    """Creates one ToolCallRecord per MCP tool call. Runs in finally — logs success AND errors.

    Only logs requests where method == "tools/call". Health checks and
    other MCP protocol messages are passed through without logging.
    """

    def __init__(self, app: Any, namespace_ctx: ContextVar[str]) -> None:
        super().__init__(app)
        self._namespace_ctx = namespace_ctx

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        body = b""
        tool_name = ""
        inputs_data: Any = {}

        try:
            body = await request.body()
            payload = json.loads(body) if body else {}
            # MCP Streamable HTTP: {"method": "tools/call", "params": {"name": "...", "arguments": {...}}}
            if payload.get("method") == "tools/call":
                tool_name = payload.get("params", {}).get("name", "unknown")
                inputs_data = payload.get("params", {}).get("arguments", {})
        except (json.JSONDecodeError, AttributeError):
            pass

        if not tool_name:
            return await call_next(request)

        start = time.monotonic()
        status: Literal["success", "error", "timeout"] = "success"
        error_code: str | None = None

        try:
            response = await call_next(request)
            return response
        except TimeoutError:
            status = "timeout"
            error_code = "TOOL_TIMEOUT"
            raise
        except Exception as exc:
            status = "error"
            error_code = type(exc).__name__
            raise
        finally:
            duration_ms = int((time.monotonic() - start) * 1000)
            tokens_in = token_in_ctx.get()
            tokens_out = token_out_ctx.get()
            record = ToolCallRecord(
                id=str(uuid.uuid4()),
                tool_name=tool_name,
                namespace=self._namespace_ctx.get(),
                session_id=session_id_ctx.get(),
                inputs_hash=hash_inputs(inputs_data),
                status=status,
                error_code=error_code,
                duration_ms=duration_ms,
                llm_tokens_in=tokens_in,
                llm_tokens_out=tokens_out,
                cost_usd=estimate_cost(tokens_in, tokens_out),
                timestamp=datetime.utcnow(),
            )
            await insert_tool_call_record(record)
