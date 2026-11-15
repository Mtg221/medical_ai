"""Database connection management, migration runner, and backup/restore utilities."""

from __future__ import annotations

import hashlib
import re
import shutil
import sqlite3
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
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

    def _reconcile_legacy_sqlite_tables(self, conn: sqlite3.Connection, schema_sql: str) -> None:
        """Non-destructively upgrade legacy SQLite tables whose CHECK constraints or columns predated current DDL."""
        checks = {
            "documents": ("'TESSERACT_OCR'",),
            "extracted_entities": ("'needs_review'", "review_reason", "source_stage"),
            "extracted_relations": ("'TREATS'",),
            "jobs": ("original_filename",),
        }
        needs_upgrade: list[str] = []
        for row in conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall():
            table = str(row["name"])
            existing_sql = str(row["sql"] or "")
            required_tokens = checks.get(table, ())
            if "__legacy_backup" in existing_sql or any(tok not in existing_sql for tok in required_tokens):
                needs_upgrade.append(table)

        if not needs_upgrade:
            return

        conn.execute("PRAGMA foreign_keys = OFF;")
        conn.execute("PRAGMA legacy_alter_table = ON;")
        try:
            for table in needs_upgrade:
                m = re.search(
                    rf"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+{table}\s*\(.*?\n\);",
                    schema_sql,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                if not m:
                    continue
                create_stmt = m.group(0)
                old_cols = {
                    str(r["name"])
                    for r in conn.execute(f"PRAGMA table_info({table})").fetchall()
                }
                backup = f"{table}__legacy_backup"
                conn.execute(f"DROP TABLE IF EXISTS {backup}")
                conn.execute(f"ALTER TABLE {table} RENAME TO {backup}")
                conn.execute(create_stmt)
                new_cols = [
                    str(r["name"])
                    for r in conn.execute(f"PRAGMA table_info({table})").fetchall()
                ]
                common = [c for c in new_cols if c in old_cols]
                cols_csv = ", ".join(common)
                conn.execute(f"INSERT INTO {table} ({cols_csv}) SELECT {cols_csv} FROM {backup}")
                conn.execute(f"DROP TABLE {backup}")
        finally:
            conn.execute("PRAGMA legacy_alter_table = OFF;")
            conn.execute("PRAGMA foreign_keys = ON;")

    def apply_migrations(self) -> dict[str, Any]:
        """Apply SQLite migrations locally and optionally run PostgreSQL migration if DATABASE_URL is set."""
        sql_bytes = self.sqlite_migration_file.read_bytes()
        schema_sql = sql_bytes.decode("utf-8")
        checksum = hashlib.sha256(sql_bytes).hexdigest()
        version = "001_initial_schema"
        now_iso = datetime.now(timezone.utc).isoformat()

        with self.transaction() as conn:
            conn.executescript(schema_sql)
            self._reconcile_legacy_sqlite_tables(conn, schema_sql)
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
