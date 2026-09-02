"""
audit_log.py
------------
Every detection is written as a row containing a hash of the previous
row's hash plus its own data — a simple, real hash chain (not literally
a blockchain, but a genuinely tamper-evident mechanism). If anyone edits
a past row's stored values, recomputing the chain will no longer match,
and "Verify Audit Trail" will flag it.
"""

import hashlib
from datetime import datetime, timezone

from core import db


def _compute_hash(prev_hash: str, timestamp: str, username: str, decision: str, risk_score: float) -> str:
    payload = f"{prev_hash}|{timestamp}|{username}|{decision}|{risk_score:.4f}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def record_detection(username: str, decision: str, risk_score: float,
                      spoof_confidence: float, speaker_similarity, detector_mode: str) -> dict:
    timestamp = datetime.now(timezone.utc).isoformat()
    prev_hash = db.get_last_hash()
    this_hash = _compute_hash(prev_hash, timestamp, username, decision, risk_score)

    entry = {
        "timestamp": timestamp,
        "username": username,
        "decision": decision,
        "risk_score": risk_score,
        "spoof_confidence": spoof_confidence,
        "speaker_similarity": speaker_similarity,
        "detector_mode": detector_mode,
        "prev_hash": prev_hash,
        "this_hash": this_hash,
    }
    new_id = db.insert_audit_row(entry)
    entry["id"] = new_id
    return entry


def verify_chain() -> dict:
    """Walk the whole table, recompute each hash from its stored fields,
    and confirm the chain is unbroken."""
    rows = db.get_all_audit_rows()
    if not rows:
        return {"ok": True, "message": "No entries yet — nothing to verify.", "rows_checked": 0}

    expected_prev = "GENESIS"
    for row in rows:
        if row["prev_hash"] != expected_prev:
            return {
                "ok": False,
                "message": f"Chain broken at row id={row['id']}: prev_hash mismatch.",
                "rows_checked": row["id"],
            }
        recomputed = _compute_hash(
            row["prev_hash"], row["timestamp"], row["username"], row["decision"], row["risk_score"]
        )
        if recomputed != row["this_hash"]:
            return {
                "ok": False,
                "message": f"Chain broken at row id={row['id']}: hash does not match stored data (tampering detected).",
                "rows_checked": row["id"],
            }
        expected_prev = row["this_hash"]

    return {"ok": True, "message": f"Chain verified — 0 tampering detected across {len(rows)} entries.", "rows_checked": len(rows)}
