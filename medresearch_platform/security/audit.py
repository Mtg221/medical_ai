"""Tamper-evident hash-chained audit logging with automatic PHI and secret scrubbing (Section 24)."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import sqlite3
from typing import Any
import uuid

FORBIDDEN_METADATA_KEYS = frozenset({
    "password",
    "password_hash",
    "secret",
    "token",
    "access_token",
    "private_key",
    "raw_text",
    "patient_name",
    "nom_patient",
    "ssn",
    "nir",
    "address",
    "adresse",
    "phone",
    "telephone",
})

GENESIS_HASH = "0" * 64


def sanitize_audit_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Scrub any potentially sensitive or forbidden keys before persisting to the audit chain."""
    if not metadata:
        return {}
    cleaned: dict[str, Any] = {}
    for key, val in metadata.items():
        key_lower = str(key).lower()
        if key_lower in FORBIDDEN_METADATA_KEYS or any(
            bad in key_lower for bad in ("password", "secret", "token", "raw_text", "patient_name")
        ):
            cleaned[key] = "[REDACTED_BY_AUDIT_GUARD]"
        elif isinstance(val, dict):
            cleaned[key] = sanitize_audit_metadata(val)
        elif isinstance(val, str) and len(val) > 256:
            cleaned[key] = f"[TRUNCATED_LENGTH_{len(val)}]"
        else:
            cleaned[key] = val
    return cleaned


def compute_audit_entry_hash(
    *,
    sequence_num: int,
    actor_user_id: str,
    institution_id: str,
    action_type: str,
    resource_type: str,
    resource_id: str,
    metadata_safe_json: str,
    prev_hash_sha256: str,
    created_at: str,
) -> str:
    canonical = "|".join([
        str(sequence_num),
        actor_user_id,
        institution_id,
        action_type,
        resource_type,
        resource_id,
        metadata_safe_json,
        prev_hash_sha256,
        created_at,
    ])
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def record_audit_event(
    conn: sqlite3.Connection,
    *,
    actor_user_id: str,
    institution_id: str,
    action_type: str,
    resource_type: str,
    resource_id: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append an immutable, hash-chained audit record inside the active transaction."""
    safe_meta = sanitize_audit_metadata(metadata)
    metadata_json = json.dumps(safe_meta, sort_keys=True, separators=(",", ":"))
    last_row = conn.execute(
        "SELECT sequence_num, entry_hash_sha256 FROM audit_logs ORDER BY sequence_num DESC LIMIT 1"
    ).fetchone()
    if last_row is None:
        next_seq = 1
        prev_hash = GENESIS_HASH
    else:
        next_seq = int(last_row["sequence_num"]) + 1
        prev_hash = str(last_row["entry_hash_sha256"])

    created_at = datetime.now(timezone.utc).isoformat()
    audit_id = f"aud_{uuid.uuid4().hex[:16]}"
    entry_hash = compute_audit_entry_hash(
        sequence_num=next_seq,
        actor_user_id=actor_user_id,
        institution_id=institution_id,
        action_type=action_type,
        resource_type=resource_type,
        resource_id=resource_id,
        metadata_safe_json=metadata_json,
        prev_hash_sha256=prev_hash,
        created_at=created_at,
    )
    conn.execute(
        """
        INSERT INTO audit_logs (
            audit_id, sequence_num, actor_user_id, institution_id, action_type,
            resource_type, resource_id, metadata_safe_json, prev_hash_sha256,
            entry_hash_sha256, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            audit_id,
            next_seq,
            actor_user_id,
            institution_id,
            action_type,
            resource_type,
            resource_id,
            metadata_json,
            prev_hash,
            entry_hash,
            created_at,
        ),
    )
    return {
        "audit_id": audit_id,
        "sequence_num": next_seq,
        "prev_hash_sha256": prev_hash,
        "entry_hash_sha256": entry_hash,
    }


def verify_audit_chain(conn: sqlite3.Connection) -> dict[str, Any]:
    """Verify cryptographic integrity of the entire audit log chain."""
    rows = conn.execute(
        "SELECT * FROM audit_logs ORDER BY sequence_num ASC"
    ).fetchall()
    expected_prev = GENESIS_HASH
    for idx, row in enumerate(rows, start=1):
        seq = int(row["sequence_num"])
        if seq != idx:
            return {
                "valid": False,
                "checked_entries": idx - 1,
                "error": f"Sequence gap at index {idx}: found sequence_num={seq}",
            }
        if str(row["prev_hash_sha256"]) != expected_prev:
            return {
                "valid": False,
                "checked_entries": idx - 1,
                "error": f"Broken prev_hash link at sequence_num={seq}",
            }
        recomputed = compute_audit_entry_hash(
            sequence_num=seq,
            actor_user_id=str(row["actor_user_id"]),
            institution_id=str(row["institution_id"]),
            action_type=str(row["action_type"]),
            resource_type=str(row["resource_type"]),
            resource_id=str(row["resource_id"]),
            metadata_safe_json=str(row["metadata_safe_json"]),
            prev_hash_sha256=str(row["prev_hash_sha256"]),
            created_at=str(row["created_at"]),
        )
        if recomputed != str(row["entry_hash_sha256"]):
            return {
                "valid": False,
                "checked_entries": idx - 1,
                "error": f"Tampered entry hash at sequence_num={seq}",
            }
        expected_prev = recomputed

    return {
        "valid": True,
        "checked_entries": len(rows),
        "head_hash_sha256": expected_prev,
    }
