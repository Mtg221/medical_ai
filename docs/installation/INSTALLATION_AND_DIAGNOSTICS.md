# Installation, Portability & Diagnostics Guide

**Specification References:** Sections 8, 34, 38, 39

---

## 1. Verified Hardware & OS Target
- **OS:** macOS Darwin (`x86_64` Intel) — tested on Darwin 25.6.0
- **CPU:** Intel(R) Core(TM) i7-9750H @ 2.60GHz (6 cores / 12 threads)
- **Memory:** 32 GB RAM (CPU-only; no GPU required)
- **Python:** `3.13.12` (compatible with Python `>=3.11`)

## 2. Native CPU-Only Installation (No Docker Required)
All required Python packages (`numpy`, `scipy`, `scikit-learn`, `pandas`, `pillow`, `pypdf`, `python-docx`, `reportlab`, `starlette`, `httpx`, `pytest`, `psutil`, `argon2-cffi`) run natively on CPU.

Run environment diagnostics:
```bash
python3 -m medresearch_platform.interfaces.cli diagnostics
```

## 3. Database Configuration (PostgreSQL 18 & Local SQLite)
- **Local / Sandbox Mode (Default):** Uses `var/db/medresearch.sqlite3` with `PRAGMA foreign_keys = ON` and WAL journaling, synchronized with `medresearch_platform/db/migrations/001_initial_schema.sqlite.sql`.
- **PostgreSQL 18 Mode:** When a PostgreSQL 18 server is running and `DATABASE_URL` is exported (e.g., `export DATABASE_URL=postgresql://postgres@127.0.0.1:5432/medresearch`), `DatabaseEngine.apply_migrations()` automatically executes `medresearch_platform/db/migrations/001_initial_schema.postgres.sql` via `psql`, enabling native PostgreSQL Row-Level Security (`CREATE POLICY institution_isolation_documents`).

## 4. Environment Variables
- `MEDRESEARCH_PROFILE`: `development` (default), `test`, `staging`, or `production`
- `MEDRESEARCH_TOKEN_SECRET`: Required in `production` (HMAC-SHA256 signing key)
- `MEDRESEARCH_VAULT_SECRET`: Required in `production` (Pseudonym derivation key)
- `MEDRESEARCH_MAX_CPU_THREADS`: Default `4`
- `MEDRESEARCH_MAX_MEMORY_MB`: Default `4096`
- `MEDRESEARCH_RANDOM_SEED`: Default `20261112`

## 5. Backup & Restore
`DatabaseEngine.backup_to(path)` creates an atomic online snapshot and returns its SHA-256 checksum. `DatabaseEngine.restore_from(path, expected_sha256)` verifies the checksum before restoring.
