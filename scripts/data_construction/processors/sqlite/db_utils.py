"""
SQLiteDataFunction
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from core import STRUCTURE_DIR


def get_db_path(epoch: int) -> Path:
    """GetepochSQLiteDataPath."""
    return STRUCTURE_DIR / f"lkml_epoch_{epoch:02d}.sqlite"


def init_db(conn: sqlite3.Connection) -> None:
    """Initialize SQLite Data."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS emails (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            epoch INTEGER,
            seq_in_epoch INTEGER,
            message_id TEXT UNIQUE,
            subject TEXT,
            from_addr TEXT,
            to_addr TEXT,
            cc TEXT,
            date_raw TEXT,
            timestamp REAL,
            in_reply_to TEXT,
            references_raw TEXT,
            list_id TEXT,
            body TEXT,
            body_length INTEGER,
            is_patch INTEGER,
            patch_version TEXT,
            patch_current INTEGER,
            patch_total INTEGER,
            is_cover INTEGER,
            patchwork_id TEXT,
            raw_headers_json TEXT,
            jsonl_path TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_emails_msgid ON emails(message_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_emails_in_reply_to ON emails(in_reply_to)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_emails_epoch_seq ON emails(epoch, seq_in_epoch)")
    conn.commit()

