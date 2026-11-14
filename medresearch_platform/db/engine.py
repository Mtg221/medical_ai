"""Database connection management, migration runner, and backup/restore utilities."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
import sqlite3
import subprocess
from typing import Any, Iterator

from medresearch_platform.config import PlatformConfig


class DatabaseEngine:
    """Unified relational engine supporting canonical PostgreSQL 18 DDL and local SQLite execution."""

    def __init__(self, config: PlatformConfig) -> None:
        self.config = config
        self.migrations_dir = Path(__file__).resolve().parent / "migrations"
        self.config.ensure_directories()

    @property
    def postgres_migration_file(self) -> Path:
        return self.migrations_dir / "001_initial_schema.postgres.sql"

    @property
    def sqlite_migration_file(self) -> Path:
        return self.migrations_dir / "001_initial_schema.sqlite.sql"

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.config.sqlite_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def apply_migrations(self) -> dict[str, Any]:
        """Apply SQLite migrations locally and optionally run PostgreSQL migration if DATABASE_URL is set."""
        sql_bytes = self.sqlite_migration_file.read_bytes()
        checksum = hashlib.sha256(sql_bytes).hexdigest()
        version = "001_initial_schema"
        now_iso = datetime.now(timezone.utc).isoformat()

        with self.transaction() as conn:
            conn.executescript(sql_bytes.decode("utf-8"))
            existing = conn.execute(
                "SELECT version, checksum_sha256 FROM schema_migrations WHERE version = ?",
                (version,),
            ).fetchone()
            if existing is None:
                conn.execute(
                    "INSERT INTO schema_migrations (version, applied_at, checksum_sha256) VALUES (?, ?, ?)",
                    (version, now_iso, checksum),
                )

        pg_status = "SKIPPED_NO_DATABASE_URL"
        if self.config.postgres_url:
            psql_bin = shutil.which("psql") or "/Library/PostgreSQL/18/bin/psql"
            if Path(psql_bin).exists():
                proc = subprocess.run(
                    [psql_bin, self.config.postgres_url, "-f", str(self.postgres_migration_file)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                pg_status = "APPLIED" if proc.returncode == 0 else f"ERROR: {proc.stderr.strip()}"
            else:
                pg_status = "PSQL_BINARY_NOT_FOUND"

        return {
            "version": version,
            "sqlite_checksum_sha256": checksum,
            "postgres_schema_sha256": hashlib.sha256(self.postgres_migration_file.read_bytes()).hexdigest(),
            "sqlite_path": str(self.config.sqlite_path),
            "postgres_status": pg_status,
        }

    def backup_to(self, backup_path: Path) -> str:
        """Create a consistent online backup of the relational database and return its SHA-256."""
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        src = self.connect()
        dst = sqlite3.connect(str(backup_path))
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()
        return hashlib.sha256(backup_path.read_bytes()).hexdigest()

    def restore_from(self, backup_path: Path, expected_sha256: str | None = None) -> None:
        """Verify checksum and restore database from backup."""
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        actual_sha256 = hashlib.sha256(backup_path.read_bytes()).hexdigest()
        if expected_sha256 and actual_sha256 != expected_sha256:
            raise ValueError(
                f"Backup integrity check failed: expected {expected_sha256}, got {actual_sha256}"
            )
        src = sqlite3.connect(str(backup_path))
        dst = self.connect()
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()
