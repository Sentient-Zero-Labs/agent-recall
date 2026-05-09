"""Recall CLI — `recall serve` and `recall status`."""

from __future__ import annotations

import argparse
import asyncio
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="recall",
        description="Recall — persistent memory layer for AI agents.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # serve
    serve = subparsers.add_parser("serve", help="Start the Recall MCP server.")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--db", default="recall.db", help="Path to SQLite database.")
    serve.add_argument("--reload", action="store_true", help="Auto-reload on code changes.")

    # status
    status = subparsers.add_parser("status", help="Show database stats.")
    status.add_argument("--db", default="recall.db")

    args = parser.parse_args()

    if args.command == "serve":
        _cmd_serve(args)
    elif args.command == "status":
        asyncio.run(_cmd_status(args))


def _cmd_serve(args: argparse.Namespace) -> None:
    import os
    os.environ.setdefault("RECALL_DB_PATH", args.db)

    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is required to serve. Install with: pip install uvicorn", file=sys.stderr)
        sys.exit(1)

    from recall.server import create_app
    app = create_app()
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


async def _cmd_status(args: argparse.Namespace) -> None:
    from recall.db.connection import set_db_path, init_db
    from pathlib import Path

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        print("Run 'recall serve' first to initialize the database.")
        return

    set_db_path(db_path)
    import aiosqlite
    async with aiosqlite.connect(db_path) as db:
        rows = await db.execute_fetchall("SELECT COUNT(*) FROM memories")
        total = rows[0][0]
        by_type = await db.execute_fetchall(
            "SELECT type, COUNT(*) FROM memories GROUP BY type"
        )
        pending = await db.execute_fetchall(
            "SELECT COUNT(*) FROM operations WHERE status = 'queued'"
        )

    print(f"Database: {db_path}")
    print(f"Total memories: {total}")
    for mem_type, count in by_type:
        print(f"  {mem_type}: {count}")
    print(f"Pending extractions: {pending[0][0]}")
