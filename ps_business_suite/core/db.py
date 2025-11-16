"""Shared database helpers."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import settings


def database_path() -> Path:
    return settings.db_path


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(database_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def connection_scope() -> Iterator[sqlite3.Connection]:
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_shared_schema() -> None:
    from . import crm, sales  # imported lazily to avoid circular deps

    with connection_scope() as conn:
        crm.init_schema(conn)
        sales.init_schema(conn)
