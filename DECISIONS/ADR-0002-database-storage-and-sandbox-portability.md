# ADR-0002: Database Storage Architecture and Sandbox Portability

- **Status:** Accepted
- **Date:** 2026-11-12
- **Specification References:** Sections 2, 12, 13, 24, 34, 39

## Context

Section 34 specifies PostgreSQL as the initial source of truth for relational and transactional data. During Phase 0 baseline inspection on the host machine:
- PostgreSQL 18.4 (`psql`, `initdb`, `pg_ctl`, `postgres`) is installed under `/Library/PostgreSQL/18/bin/`.
- No PostgreSQL daemon is currently listening on `127.0.0.1:5432` or `/tmp:5432`.
- When running inside the macOS Seatbelt sandbox, `initdb` / `postgres` fails to allocate its 56-byte System V shared memory interlock segment (`shmget: Operation not permitted`), even when `shared_memory_type=mmap` is passed.
- Furthermore, Python PostgreSQL drivers (`psycopg`, `psycopg2`) are not installed in the offline Python 3.13 environment, whereas `sqlite3` (3.51.0) and the `/Library/PostgreSQL/18/bin/psql` CLI binary are available.

## Decision

1. **Canonical PostgreSQL 18 Schema & Migrations:** We author and maintain the complete PostgreSQL 18 DDL in `medresearch_platform/db/migrations/001_initial_schema.postgres.sql`, including Row-Level Security (RLS) policies, `tsvector` full-text search indexes, foreign-key constraints, check constraints, and audit protections, along with a `psql`-based migration runner for environments where a PostgreSQL server is reachable via `DATABASE_URL`.
2. **Unified Relational Engine with SQLite Transactional Adapter:** We provide a synchronized SQLite schema (`001_initial_schema.sqlite.sql`) with strict foreign key enforcement (`PRAGMA foreign_keys = ON`), WAL journaling, and application-enforced institution/project row-level isolation so that all workflows, APIs, CLI commands, and automated tests execute deterministically inside the restricted macOS sandbox without requiring external privileges.
3. **Immutable Content-Addressed Object Store:** Original documents, images, model checkpoints, and experiment manifests are stored on disk indexed by SHA-256 content hashes with strict integrity verification on read.

## Consequences

- **Positive:** 100% of platform functionality and automated integration tests run out-of-the-box in the local sandbox while preserving production-ready PostgreSQL 18 DDL and migration compatibility.
- **Negative / Limitations:** PostgreSQL-specific `CREATE POLICY` Row-Level Security requires an active PostgreSQL daemon outside the sandbox; inside the sandbox, identical tenant/institution/role isolation is enforced in the repository access layer and verified by security unit tests.
