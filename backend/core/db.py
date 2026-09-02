"""
db.py
-----
SQLite storage for enrolled voice profiles and the audit log. Zero
setup — the file is created automatically on first run.
"""

import os
import sqlite3
from datetime import datetime, timezone

import numpy as np

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "voicesentinel.db")


def get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS enrolled_voices (
            username TEXT PRIMARY KEY,
            embedding BLOB NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            username TEXT NOT NULL,
            decision TEXT NOT NULL,
            risk_score REAL NOT NULL,
            spoof_confidence REAL NOT NULL,
            speaker_similarity REAL,
            detector_mode TEXT NOT NULL,
            prev_hash TEXT NOT NULL,
            this_hash TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def save_enrollment(username: str, embedding: np.ndarray) -> None:
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO enrolled_voices (username, embedding, created_at)
        VALUES (?, ?, ?)
        ON CONFLICT(username) DO UPDATE SET embedding=excluded.embedding,
                                             created_at=excluded.created_at
        """,
        (username, embedding.astype(np.float32).tobytes(), datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


def get_embedding(username: str):
    conn = get_connection()
    row = conn.execute(
        "SELECT embedding FROM enrolled_voices WHERE username = ?", (username,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return np.frombuffer(row["embedding"], dtype=np.float32)


def list_users():
    conn = get_connection()
    rows = conn.execute("SELECT username FROM enrolled_voices ORDER BY username").fetchall()
    conn.close()
    return [r["username"] for r in rows]


def get_last_hash() -> str:
    conn = get_connection()
    row = conn.execute("SELECT this_hash FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    return row["this_hash"] if row else "GENESIS"


def insert_audit_row(entry: dict) -> int:
    """Inserts a row and returns its new integer id."""
    conn = get_connection()
    cur = conn.execute(
        """
        INSERT INTO audit_log
            (timestamp, username, decision, risk_score, spoof_confidence,
             speaker_similarity, detector_mode, prev_hash, this_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            entry["timestamp"],
            entry["username"],
            entry["decision"],
            entry["risk_score"],
            entry["spoof_confidence"],
            entry["speaker_similarity"],
            entry["detector_mode"],
            entry["prev_hash"],
            entry["this_hash"],
        ),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def get_all_audit_rows():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM audit_log ORDER BY id ASC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_audit_row(row_id: int):
    conn = get_connection()
    row = conn.execute("SELECT * FROM audit_log WHERE id = ?", (row_id,)).fetchone()
    conn.close()
    return dict(row) if row else None
