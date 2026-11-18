"""Platform configuration, environment profiles, and CPU-first resource limits."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

EnvironmentProfile = Literal["development", "test", "staging", "production"]

REQUIRED_STATUS_VOCABULARY: tuple[str, ...] = (
    "Planned",
    "In progress",
    "Implemented",
    "Software-tested",
    "Experimentally validated",
    "Independently replicated",
    "Clinically validated",
    "Blocked",
    "Deferred",
    "Not supported",
)


@dataclass(frozen=True)
class PlatformConfig:
    """Immutable runtime configuration for the medical research platform."""

    profile: EnvironmentProfile
    workspace_dir: Path
    storage_dir: Path
    sqlite_path: Path
    postgres_url: str | None
    max_cpu_threads: int
    max_memory_mb: int
    max_upload_bytes: int
    max_document_pages: int
    default_random_seed: int
    token_secret_key: str = field(repr=False)
    pseudonym_vault_key: str = field(repr=False)
    max_document_chars: int = 500_000
    job_timeout_seconds: float = 120.0
    job_timeout_per_100k_bytes_seconds: float = 90.0

    def compute_document_job_timeout(self, payload_bytes: int) -> float:
        """Derive document job timeout from base timeout plus document byte length."""
        extra = (max(0, int(payload_bytes)) / 100_000.0) * float(self.job_timeout_per_100k_bytes_seconds)
        return float(self.job_timeout_seconds) + extra

    @classmethod
    def from_env(cls, workspace_dir: Path | str | None = None) -> PlatformConfig:
        base_dir = Path(
            workspace_dir
            or os.environ.get("MEDRESEARCH_WORKSPACE_DIR")
            or Path(__file__).resolve().parent.parent
        ).resolve()

        raw_profile = os.environ.get("MEDRESEARCH_PROFILE", "development").strip().lower()
        if raw_profile not in ("development", "test", "staging", "production"):
            raise ValueError(f"Unsupported MEDRESEARCH_PROFILE: {raw_profile}")
        profile: EnvironmentProfile = raw_profile  # type: ignore[assignment]

        storage_dir = Path(
            os.environ.get("MEDRESEARCH_STORAGE_DIR", str(base_dir / "var" / "storage"))
        ).resolve()
        db_env = os.environ.get("MEDRESEARCH_DB_PATH") or os.environ.get("MEDRESEARCH_SQLITE_PATH")
        sqlite_path = Path(
            db_env or str(base_dir / "var" / "db" / "medresearch.sqlite3")
        ).resolve()
        postgres_url = os.environ.get("DATABASE_URL") or os.environ.get("MEDRESEARCH_POSTGRES_URL")

        max_cpu_threads = int(os.environ.get("MEDRESEARCH_MAX_CPU_THREADS", "4"))
        max_memory_mb = int(os.environ.get("MEDRESEARCH_MAX_MEMORY_MB", "4096"))
        max_upload_bytes = int(os.environ.get("MEDRESEARCH_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
        max_document_pages = int(os.environ.get("MEDRESEARCH_MAX_DOCUMENT_PAGES", "200"))
        default_random_seed = int(os.environ.get("MEDRESEARCH_RANDOM_SEED", "20261112"))

        token_secret = os.environ.get("MEDRESEARCH_TOKEN_SECRET")
        vault_secret = os.environ.get("MEDRESEARCH_VAULT_SECRET")

        if profile == "production" and (not token_secret or not vault_secret):
            raise RuntimeError(
                "MEDRESEARCH_TOKEN_SECRET and MEDRESEARCH_VAULT_SECRET must be set explicitly in production."
            )

        if not token_secret:
            dev_secret_file = storage_dir / ".dev_token_secret"
            if dev_secret_file.exists():
                token_secret = dev_secret_file.read_text(encoding="utf-8").strip()
            else:
                token_secret = secrets.token_hex(32)
                dev_secret_file.parent.mkdir(parents=True, exist_ok=True)
                dev_secret_file.write_text(token_secret, encoding="utf-8")
                dev_secret_file.chmod(0o600)
        if not vault_secret:
            vault_secret = secrets.token_hex(32)

        return cls(
            profile=profile,
            workspace_dir=base_dir,
            storage_dir=storage_dir,
            sqlite_path=sqlite_path,
            postgres_url=postgres_url,
            max_cpu_threads=max_cpu_threads,
            max_memory_mb=max_memory_mb,
            max_upload_bytes=max_upload_bytes,
            max_document_pages=max_document_pages,
            default_random_seed=default_random_seed,
            token_secret_key=token_secret,
            pseudonym_vault_key=vault_secret,
            max_document_chars=int(os.environ.get("MEDRESEARCH_MAX_DOCUMENT_CHARS", "500000")),
            job_timeout_seconds=float(os.environ.get("MEDRESEARCH_JOB_TIMEOUT_SECONDS", "120.0")),
            job_timeout_per_100k_bytes_seconds=float(
                os.environ.get("MEDRESEARCH_JOB_TIMEOUT_PER_100K_BYTES_SECONDS", "90.0")
            ),
        )

    def ensure_directories(self) -> None:
        """Create required runtime directories with restrictive permissions."""
        for folder in (
            self.storage_dir,
            self.storage_dir / "originals",
            self.storage_dir / "derived",
            self.storage_dir / "checkpoints",
            self.storage_dir / "experiments",
            self.storage_dir / "datasets",
            self.sqlite_path.parent,
        ):
            folder.mkdir(parents=True, exist_ok=True)
