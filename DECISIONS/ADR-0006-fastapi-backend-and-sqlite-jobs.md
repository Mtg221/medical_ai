# ADR-0006: FastAPI + Pydantic v2 Backend and SQLite-Backed Job Queue

- **Status:** Accepted
- **Date:** 2026-11-14
- **References:** `docs/MASTER_PROMPT.md` Part C.1, Phase 6

## Context

The current API is built on raw Starlette with manual JSON parsing and synchronous request handling. Moving to a single source-of-truth extraction contract (Pydantic → JSON Schema → TypeScript types) and supporting multi-file upload with live SSE progress requires typed request/response models and background job tracking.

## Decision

1. **FastAPI + Pydantic v2:** Migrate the API layer (Phase 6) from raw Starlette to FastAPI (built on Starlette, preserving existing ASGI/auth/security middleware) with Pydantic v2 models as the single source of truth for `schema/medical_extraction.schema.json`.
2. **SQLite-Backed Background Jobs:** Use FastAPI `BackgroundTasks` backed by a SQLite `jobs` table (status, progress %, current stage, error) and SSE (`GET /api/v1/jobs/{id}/events`), avoiding Redis/Celery operational overhead on a single local machine.

## Consequences

- **Positive:** Automatic OpenAPI 3.1 schema, generated TypeScript types for the React frontend, zero external broker dependencies.
- **Negative / Limitations:** Single-process concurrency; suitable for local research workstation deployment rather than multi-node horizontal scaling.
