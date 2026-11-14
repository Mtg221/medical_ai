"""Authentication, Argon2id password verification, signed tokens, and RBAC/ABAC authorization."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
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


@dataclass(frozen=True)
class AuthenticatedPrincipal:
    user_id: str
    username: str
    institution_id: str
    role: str
    permissions: frozenset[str]
    expires_at_iso: str

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
        raise PermissionError("Malformed access token.")
    payload_b64, provided_sig = parts
    expected_sig = hmac.new(
        secret_key.encode("utf-8"), payload_b64.encode("ascii"), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(provided_sig, expected_sig):
        raise PermissionError("Invalid access token signature.")
    try:
        raw_json = base64.urlsafe_b64decode(payload_b64.encode("ascii")).decode("utf-8")
        payload = json.loads(raw_json)
    except Exception as exc:
        raise PermissionError("Corrupted access token payload.") from exc

    exp_dt = datetime.fromisoformat(payload["exp"])
    if datetime.now(timezone.utc) >= exp_dt:
        raise PermissionError("Access token has expired.")

    return AuthenticatedPrincipal(
        user_id=str(payload["sub"]),
        username=str(payload["usr"]),
        institution_id=str(payload["inst"]),
        role=str(payload["role"]),
        permissions=frozenset(payload["perms"]),
        expires_at_iso=str(payload["exp"]),
    )
