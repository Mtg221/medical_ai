# Project Status & Requirement Traceability Matrix

**Last Updated:** 2026-11-12T23:47:00Z  
**Current Milestone Completed:** Phase 0 (Discovery & Foundation) + Phase 1 (End-to-End CPU-Only Medical Document Extraction & Random-Init Neural Baseline) + Foundational Capabilities for Phases 2–8.

---

## 1. Verified Machine & Environment Baseline

- **Initial Repository State:** Empty directory containing only `MASTER_SPECIFICATION_AND_IMPLEMENTATION_PROMPT.md` (50,159 bytes).
- **Operating System:** macOS Darwin 25.6.0 (`x86_64`)
- **CPU:** Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz (6 physical cores / 12 logical threads)
- **Memory:** 32.0 GB RAM (34,359,738,368 bytes total; ~16.5 GB available)
- **GPU:** None (`CPU-ONLY (Native)` execution mode enforced)
- **Storage:** 465.63 GB total, ~234.62 GB free on `/System/Volumes/Data`
- **Installed Runtimes & Tools:**
  - Python `3.13.12` (`numpy 2.4.2`, `scipy 1.17.1`, `scikit-learn 1.9.1`, `pandas 3.0.5`, `pillow 12.1.1`, `pypdf 6.19.0`, `python-docx 1.2.0`, `reportlab 5.0.0`, `starlette 1.6.0`, `httpx 0.28.1`, `pytest 9.1.1`, `psutil 7.2.2`, `argon2-cffi 25.1.0`)
  - Node.js `v24.18.0`, npm `11.16.0`
  - PostgreSQL `18.4` (`psql`, `initdb`, `pg_ctl` at `/Library/PostgreSQL/18/bin/`)
  - SQLite `3.51.0`
- **Sandbox Constraints Recorded:**
  - Starting a new PostgreSQL daemon inside the macOS Seatbelt sandbox fails on `shmget(size=56): Operation not permitted`. Canonical PostgreSQL 18 DDL (`001_initial_schema.postgres.sql`) is provided alongside a synchronized transactional SQLite WAL adapter (`001_initial_schema.sqlite.sql`) for 100% offline/sandbox execution (see `DECISIONS/ADR-0002-database-storage-and-sandbox-portability.md`).

---

## 2. Requirement Traceability Matrix (Using Section 47 Required Status Vocabulary)

Allowed Vocabulary: `Planned`, `In progress`, `Implemented`, `Software-tested`, `Experimentally validated`, `Independently replicated`, `Clinically validated`, `Blocked`, `Deferred`, `Not supported`.

| Spec Section | Capability / Requirement | Status | Code Module | Verification / Test Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **Sec 6.1 & 9** | Medical Document / Native PDF, DOCX, TXT, Image Parsing & Sectioning | **Software-tested** | `medresearch_platform/documents/parsers.py` | `tests/test_documents_and_splits.py` |
| **Sec 6.1 & 7** | Custom Random-Init OCR Neural Network (`CustomOCRConvNet-v1`) + CER/WER | **Experimentally validated** *(Synthetic Benchmark)* | `medresearch_platform/ai/ocr_model.py` | `tests/test_ai_models_and_evaluation.py` (`CER=0.0` clean, `0.1895` noisy) |
| **Sec 6.2 & 10** | Custom Random-Init Medical NLP (`CustomMedicalNLPNet-v1`): Entities, Assertions, Temporality, Relations (EN/FR) | **Experimentally validated** *(Synthetic Benchmark)* | `medresearch_platform/ai/nlp_model.py` | `tests/test_ai_models_and_evaluation.py` (`Exact F1=0.9610`, `Assertion Acc=0.9730`) |
| **Sec 6.3 & 11** | Medical Computer Vision Quality, Modality & ROI Baseline (`CustomMedicalVisionNet-v1`) | **Software-tested** | `medresearch_platform/ai/vision_multimodal.py` | `tests/test_ai_models_and_evaluation.py` |
| **Sec 6.4 & 11** | Multimodal Fusion & Disagreement Analyzer (`CustomMultimodalFusionNet-v1`, Verified Pairs Only) | **Software-tested** | `medresearch_platform/ai/vision_multimodal.py` | `tests/test_ai_models_and_evaluation.py` |
| **Sec 6.5 & 12** | Longitudinal Timelines, Cross-Document Contradiction Detection & Knowledge Graph | **Software-tested** | `medresearch_platform/ai/longitudinal.py` | `tests/test_api_and_cli.py` |
| **Sec 6.6 & 13** | Hybrid Lexical + Structured Terminology + Custom Neural Retrieval (`Precision@k`, `Recall@k`, `MRR`, `nDCG`) | **Experimentally validated** *(Synthetic Benchmark)* | `medresearch_platform/ai/longitudinal.py` | `exp_phase1_baseline_v1_manifest.json` (`MRR=1.0`, `nDCG@3=0.7654`) |
| **Sec 7** | Cryptographic Random Weight Initialization & Provenance Verification (`initial_weights_sha256`) | **Software-tested** | `medresearch_platform/ai/random_init.py` | `tests/test_ai_models_and_evaluation.py` |
| **Sec 14** | Versioned Bilingual (EN/FR) Terminology Registry (ICD-10, SNOMED-CT, LOINC, RxNorm, ATC) | **Software-tested** | `medresearch_platform/ai/terminology.py` | `tests/test_ai_models_and_evaluation.py` |
| **Sec 15 & 17** | Dataset Registry, License/Consent Metadata & Patient/Family Split Leakage Prevention | **Software-tested** | `medresearch_platform/documents/synthetic_generator.py` | `tests/test_documents_and_splits.py` |
| **Sec 16** | Human Review, Correction, Inter-Annotator Agreement (`Cohen's kappa`) & Adjudication | **Software-tested** | `medresearch_platform/workflows/review.py` | `tests/test_ai_models_and_evaluation.py`, `tests/test_api_and_cli.py` |
| **Sec 18, 20, 21** | Calibration (`ECE`, `Brier`), Selective Prediction Coverage-Risk, Bootstrap 95% CIs & Subgroup Fairness | **Experimentally validated** *(Synthetic Benchmark)* | `medresearch_platform/ai/evaluation.py` | `exp_phase1_baseline_v1_manifest.json` |
| **Sec 24 & 25** | Argon2id Auth, HMAC Scoped Tokens, RBAC/ABAC, Bilingual PII Scrubber, Date-Shifting, Pseudonym Vault & Tamper-Evident Audit Chain | **Software-tested** | `medresearch_platform/security/` | `tests/test_security_privacy_audit.py` |
| **Sec 27–30** | Multi-Node Federated Simulation (`FedAvg`/`FedProx`, Byzantine Outlier Quarantine, DP $\epsilon$ Accounting) | **Experimentally validated** *(Simulated Multi-Node)* | `medresearch_platform/ai/evaluation.py` | `tests/test_ai_models_and_evaluation.py` |
| **v40.0 Part 3 (§6–§12)** | Six Independent Medical Extraction Approaches (Rule-Based, TF-IDF+ML, BiLSTM, BiLSTM-CRF, Random-Init Transformer, 7-Stage Hybrid Multimodal) & Common Output Contract | **Software-tested** | `medresearch_platform/extraction/` | `tests/test_six_extraction_approaches.py` (gradient checks, BIO constraints, contract, no-false-pairs, CPU benchmark) |
| **Dermatology CV & Multimodal** | Streaming Metadata Catalog (573k CloudFront corpus schema), Leak-Free Group Splits & Pure-NumPy Random-Init `RandomInitDermConvNet` with Skin-Tone/Modality Fairness | **Software-tested** | `medresearch_platform/dermatology/` | `tests/test_dermatology_cv.py` (gzip streaming, split integrity, Conv2D finite-diff check, synthetic fairness eval) |
| **Sec 33 & 35** | Versioned REST API (`/api/v1/...`), OpenAPI 3.1, CLI & Accessible Bilingual Web Workspace | **Software-tested** | `medresearch_platform/interfaces/` | `tests/test_api_and_cli.py` |
| **Sec 34 & 38** | Canonical PostgreSQL 18 DDL + Transactional SQLite WAL + Online Backup/Restore Verification | **Software-tested** | `medresearch_platform/db/` | `tests/test_documents_and_splits.py` |
| **Sec 28** | Live Multi-Institution Networked Federated Deployment across External Hospitals | **Planned** *(Requires Institutional Agreements & Network Approval)* | — | — |
| **Sec 40–43** | Live External Literature Surveillance APIs & Automated CONSORT-AI/TRIPOD+AI Manuscript Export | **Planned** *(Phase 9)* | — | — |
| **Sec 5 & 47** | Autonomous Clinical Diagnosis, Treatment Prescribing, or Clinical Validation Claims | **Not supported** *(Explicitly Prohibited by Specification)* | — | — |
