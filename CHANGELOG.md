# Changelog

All notable changes to the Medical Information Intelligence & Federated Scientific Research Platform are documented in this file.

## [0.1.0] - 2026-11-12

### Added
- **Phase 0 Foundation:**
  - Hardware, OS, and dependency diagnostic inspector (`medresearch_platform/diagnostics.py`).
  - Architecture Decision Records `ADR-0001` through `ADR-0004` in `DECISIONS/`.
  - Canonical PostgreSQL 18 schema (`001_initial_schema.postgres.sql`) with Row-Level Security (RLS) and synchronized local SQLite WAL engine (`001_initial_schema.sqlite.sql`) with online backup/restore verification.
- **Zero-Trust Security & Privacy:**
  - Argon2id password hashing, HMAC-SHA256 scoped access tokens, and RBAC/ABAC permission enforcement (`medresearch_platform/security/auth.py`).
  - Bilingual (English/French) PII detector, interval-preserving subject date shifter, and isolated reversible pseudonym vault (`medresearch_platform/security/privacy.py`).
  - Untrusted document byte inspector, DOCX zip-bomb guard, and prompt-injection pattern flagger (`medresearch_platform/security/safety.py`).
  - Tamper-evident SHA-256 hash-chained audit log with automatic PHI/secret scrubbing (`medresearch_platform/security/audit.py`).
- **Phase 1 Medical Document Intelligence & Primary Random-Init AI Track:**
  - Cryptographic random weight initialization and SHA-256 weight-origin verifier (`medresearch_platform/ai/random_init.py`).
  - Custom 2D Patch-Conv + MLP OCR Neural Network (`CustomOCRConvNet-v1`, 41,461 parameters) trained from random initialization with AdamW analytical backpropagation (`medresearch_platform/ai/ocr_model.py`).
  - Custom Multi-Head Medical NLP Neural Network (`CustomMedicalNLPNet-v1`, 7,703 parameters) extracting bilingual entities, assertions (`AFFIRMED`, `NEGATED`, `UNCERTAIN`, `HYPOTHETICAL`, `FAMILY_HISTORY`), temporality (`CURRENT`, `HISTORICAL`, `PLANNED`), medication states, and relations (`medresearch_platform/ai/nlp_model.py`).
  - Versioned Bilingual Terminology Registry (`BilingualTerminologyRegistry`) covering ICD-10, SNOMED-CT, LOINC, RxNorm, and ATC (`medresearch_platform/ai/terminology.py`).
  - Vision quality/ROI baseline (`CustomMedicalVisionNet-v1`) and verified-pair Multimodal Disagreement Analyzer (`CustomMultimodalFusionNet-v1`) (`medresearch_platform/ai/vision_multimodal.py`).
  - Longitudinal subject timeline builder, cross-document contradiction detector, and hybrid retrieval scorer (`medresearch_platform/ai/longitudinal.py`).
  - Calibration (`ECE`, `Brier`), selective prediction coverage-risk curves, bootstrap 95% CIs, English/French subgroup fairness, and `FedAvg`/`FedProx` federated simulation with Byzantine outlier quarantine (`medresearch_platform/ai/evaluation.py`).
- **Interfaces & Verification:**
  - Starlette REST API (`/api/v1/...`), OpenAPI 3.1 schema, Accessible Bilingual Web Workspace (`/`), and CLI (`medresearch_platform/interfaces/`).
  - 17 automated unit, security, scientific, and end-to-end integration tests (`tests/`).
