"""Authentication, Argon2id password verification, signed tokens, and RBAC/ABAC authorization."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

ROLE_DEFAULT_PERMISSIONS: dict[str, tuple[str, ...]] = {
    "RESEARCHER": (
        "medical_data:read",
        "dataset:read",
        "experiment:read",
        "terminology:read",
        "search:query",
    ),
    "ANNOTATOR": (
        "medical_data:read",
        "annotation:review",
        "terminology:read",
    ),
    "ADJUDICATOR": (
        "medical_data:read",
        "annotation:review",
        "annotation:adjudicate",
        "terminology:read",
    ),
    "ML_ENGINEER": (
        "dataset:read",
        "training:execute",
        "experiment:read",
        "model:register",
        "terminology:read",
    ),
    "INSTITUTION_ADMIN": (
        "medical_data:read",
        "medical_data:write",
        "dataset:read",
        "dataset:write",
        "annotation:review",
        "annotation:adjudicate",
        "training:execute",
        "experiment:read",
        "model:register",
        "federated:coordinate",
        "terminology:read",
        "search:query",
        "admin:govern",
        "audit:read",
    ),
    "AUDITOR": (
        "audit:read",
        "dataset:read",
        "experiment:read",
    ),
}

_PASSWORD_HASHER = PasswordHasher(time_cost=2, memory_cost=19456, parallelism=2)


class AuthenticationError(PermissionError):
    """Raised when a Bearer token is missing, malformed, invalid, or expired (HTTP 401)."""


@dataclass(frozen=True)
class AuthenticatedPrincipal:
    user_id: str
    username: str
    institution_id: str
    role: str
    permissions: frozenset[str]
    expires_at_iso: str
    jti: str = ""

    def has_permission(self, permission: str) -> bool:
        return permission in self.permissions

    def assert_permission(self, permission: str) -> None:
        if not self.has_permission(permission):
            raise PermissionError(
                f"Principal '{self.user_id}' (role={self.role}) lacks required permission '{permission}'."
            )

    def assert_institution_access(self, target_institution_id: str) -> None:
        if self.institution_id != target_institution_id:
            raise PermissionError(
                f"Cross-institution access denied: principal belongs to '{self.institution_id}', "
                f"target belongs to '{target_institution_id}'."
            )


def hash_password(plain_password: str) -> str:
    if len(plain_password) < 8:
        raise ValueError("Password must be at least 8 characters long.")
    return _PASSWORD_HASHER.hash(plain_password)


def verify_password(password_hash: str, plain_password: str) -> bool:
    try:
        return bool(_PASSWORD_HASHER.verify(password_hash, plain_password))
    except VerifyMismatchError:
        return False


def issue_access_token(
    *,
    user_id: str,
    username: str,
    institution_id: str,
    role: str,
    permissions: Iterable[str],
    secret_key: str,
    ttl_minutes: int = 120,
) -> str:
    exp = (datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)).isoformat()
    payload = {
        "jti": uuid.uuid4().hex,
        "sub": user_id,
        "usr": username,
        "inst": institution_id,
        "role": role,
        "perms": sorted(set(permissions)),
        "exp": exp,
    }
    payload_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload_b64 = base64.urlsafe_b64encode(payload_bytes).decode("ascii")
    sig = hmac.new(secret_key.encode("utf-8"), payload_b64.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{payload_b64}.{sig}"


def verify_access_token(token: str, secret_key: str) -> AuthenticatedPrincipal:
    parts = token.split(".")
    if len(parts) != 2:
        raise AuthenticationError("Malformed access token.")
    payload_b64, provided_sig = parts
    expected_sig = hmac.new(
        secret_key.encode("utf-8"), payload_b64.encode("ascii"), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(provided_sig, expected_sig):
        raise AuthenticationError("Invalid access token signature.")
    try:
        raw_json = base64.urlsafe_b64decode(payload_b64.encode("ascii")).decode("utf-8")
        payload = json.loads(raw_json)
    except Exception as exc:
        raise AuthenticationError("Corrupted access token payload.") from exc

    exp_dt = datetime.fromisoformat(payload["exp"])
    if datetime.now(timezone.utc) >= exp_dt:
        raise AuthenticationError("Access token has expired.")

    return AuthenticatedPrincipal(
        user_id=str(payload["sub"]),
        username=str(payload["usr"]),
        institution_id=str(payload["inst"]),
        role=str(payload["role"]),
        permissions=frozenset(payload["perms"]),
        expires_at_iso=str(payload["exp"]),
        jti=str(payload.get("jti", "")),
    )


class LoginRateLimiter:
    """Thread-safe sliding-window rate limiter to mitigate login brute-force attacks."""

    def __init__(self, *, max_attempts: int = 10, window_seconds: int = 300) -> None:
        self.max_attempts = max(1, int(max_attempts))
        self.window_seconds = max(1, int(window_seconds))
        self._failures: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def _prune(self, dq: deque[float], now: float) -> None:
        cutoff = now - self.window_seconds
        while dq and dq[0] <= cutoff:
            dq.popleft()

    def check(self, key: str, *, now: float | None = None) -> tuple[bool, int]:
        ts = time.monotonic() if now is None else now
        with self._lock:
            dq = self._failures.get(key)
            if not dq:
                return True, 0
            self._prune(dq, ts)
            if len(dq) >= self.max_attempts:
                retry_after = max(1, int(self.window_seconds - (ts - dq[0])) + 1)
                return False, retry_after
            return True, 0

    def record_failure(self, key: str, *, now: float | None = None) -> None:
        ts = time.monotonic() if now is None else now
        with self._lock:
            dq = self._failures.setdefault(key, deque())
            self._prune(dq, ts)
            dq.append(ts)

    def reset(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)


class SQLiteLoginRateLimiter:
    """Multi-worker-safe sliding-window login rate limiter backed by SQLite `login_attempts` (SEC-002)."""

    def __init__(self, db: object, *, max_attempts: int = 10, window_seconds: int = 300) -> None:
        self._db = db
        self.max_attempts = max(1, int(max_attempts))
        self.window_seconds = max(1, int(window_seconds))

    def check(self, key: str, *, now: float | None = None) -> tuple[bool, int]:
        ts = time.time() if now is None else now
        cutoff = ts - self.window_seconds
        with self._db.transaction() as conn:  # type: ignore[attr-defined]
            conn.execute("DELETE FROM login_attempts WHERE rate_key = ? AND attempted_at <= ?", (key, cutoff))
            rows = conn.execute(
                "SELECT attempted_at FROM login_attempts WHERE rate_key = ? ORDER BY attempted_at ASC",
                (key,),
            ).fetchall()
            if len(rows) >= self.max_attempts:
                oldest = float(rows[0]["attempted_at"])
                retry_after = max(1, int(self.window_seconds - (ts - oldest)) + 1)
                return False, retry_after
            return True, 0

    def record_failure(self, key: str, *, now: float | None = None) -> None:
        ts = time.time() if now is None else now
        with self._db.transaction() as conn:  # type: ignore[attr-defined]
            conn.execute("INSERT INTO login_attempts (rate_key, attempted_at) VALUES (?, ?)", (key, ts))

    def reset(self, key: str) -> None:
        with self._db.transaction() as conn:  # type: ignore[attr-defined]
            conn.execute("DELETE FROM login_attempts WHERE rate_key = ?", (key,))
