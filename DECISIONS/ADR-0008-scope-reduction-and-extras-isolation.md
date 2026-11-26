# ADR-0008: Scope Reduction and Isolation of Non-Core Modules to `extras/`

- **Status:** Accepted
- **Date:** 2026-11-14
- **References:** `docs/MASTER_PROMPT.md` Parts A.3, C.2, Phase 0; `docs/AUDIT.md`

## Context

The repository accumulated features outside the core bilingual medical information extraction mission: dermatology computer vision (`medresearch_platform/dermatology/`), general medical image modality/ROI and multimodal fusion (`medresearch_platform/ai/vision_multimodal.py`), and federated learning simulation. These dilute focus, complicate the API/UI, and add unvalidated claims.

## Decision

1. **Isolate Non-Core Code in `extras/`:** Move `medresearch_platform/dermatology/`, `medresearch_platform/ai/vision_multimodal.py`, and `tests/test_dermatology_cv.py` into `extras/`, frozen and excluded from the core extraction pipeline and default `pytest` run. Nothing is deleted.
2. **Core Focus:** Concentrate exclusively on bilingual (EN/FR) medical document parsing/OCR, 6-approach NER + assertion + relation + temporal extraction, scientific evaluation, FastAPI backend, React/TS workspace, and traceable EN/FR reports.

## Consequences

- **Positive:** Smaller, cleaner core surface area; faster test runs; clearer architecture and honest documentation.
- **Negative / Limitations:** Dermatology benchmark and vision-modality API endpoints are removed from the active API (archived in `extras/`).
