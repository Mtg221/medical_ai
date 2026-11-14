# Implementation Roadmap & Phased Delivery Plan

**Platform:** Medical Information Intelligence & Federated Scientific Research Platform  
**Target Hardware:** Intel Core i7-9750H (6C/12T), 32 GB RAM, CPU-only

---

## Phase 0 — Discovery and Foundation (**Completed — Software-tested**)
- **Deliverables:** Hardware/OS/runtime baseline inspection, Architecture Decision Records (`DECISIONS/ADR-0001` to `ADR-0004`), privacy-safe bilingual synthetic data policy, environment configuration, and diagnostic readiness reporter (`medresearch_platform.diagnostics`).
- **Acceptance Criteria:** Verified baseline report, reproducible CLI diagnostics command, zero hardcoded secrets.

## Phase 1 — Minimal End-to-End Document Extraction (**Completed — Experimentally validated on Synthetic Benchmark**)
- **Deliverables:**
  - Canonical PostgreSQL 18 schema (`001_initial_schema.postgres.sql`) + local transactional SQLite WAL engine (`001_initial_schema.sqlite.sql`).
  - Content-addressed immutable document storage (`SHA-256`) + untrusted input & zip-bomb/prompt-injection guard.
  - Native PDF, DOCX, TXT, and PNG parsing with clinical section segmentation.
  - Custom random-initialized 2D Patch-Conv + MLP OCR model (`CustomOCRConvNet-v1`) trained via AdamW analytical backpropagation (`CER = 0.0` on clean synthetic scans, `0.1895` under Gaussian noise perturbation).
  - Custom random-initialized Multi-Head Medical NLP model (`CustomMedicalNLPNet-v1`) for bilingual (EN/FR) entity, assertion (`AFFIRMED`, `NEGATED`, `UNCERTAIN`, `HYPOTHETICAL`, `FAMILY_HISTORY`), temporality (`CURRENT`, `HISTORICAL`, `PLANNED`), medication status, and relation extraction (`Exact F1 = 0.9610`).
  - Human review & correction API/workflow (`submit_entity_human_review`) and reproducible CPU-only experiment manifest (`exp_phase1_baseline_v1_manifest.json`).

## Phase 2 — Dataset and Annotation Lifecycle (**Core Implemented & Software-tested; Advanced Blind UI Planned**)
- **Deliverables:** Dataset registry with license/consent/de-identification metadata, patient and document-family leak-free split verification (`verify_leak_free_splits`), inter-annotator agreement (`Cohen's kappa`), and immutable review audit trails.
- **Next Expansion:** Multi-round blind adjudication assignment queues and specialized visual bounding-box annotation canvas.

## Phase 3 — Medical NLP and Terminology (**Core Implemented & Experimentally validated on Synthetic Corpus**)
- **Deliverables:** Versioned English/French terminology registry (`BilingualTerminologyRegistry`) mapping to ICD-10, SNOMED-CT, LOINC, RxNorm, and ATC while preserving `AMBIGUOUS` and `UNMAPPED` mentions; bilingual assertion and temporal extraction.
- **Next Expansion:** Coreference chain resolution across multi-page discharge summaries and licensed UMLS/SNOMED full edition ingestion (subject to institutional licensing).

## Phase 4 — Research Laboratory and Benchmarking (**Core Implemented & Experimentally validated**)
- **Deliverables:** Model registry with cryptographic random-init provenance verification (`initial_weights_sha256`), Expected Calibration Error (`ECE`), Brier score, reliability diagrams, selective prediction coverage-risk curves, bootstrap 95% confidence intervals, English/French subgroup fairness evaluation, and CPU/RAM telemetry.
- **Next Expansion:** Multi-seed cross-validation orchestration and automated hyperparameter grid sweeps.

## Phase 5 — Knowledge and Retrieval (**Core Implemented & Software-tested**)
- **Deliverables:** Longitudinal subject timeline builder, cross-document contradiction detector, `EXTRACTED` vs `INFERRED` relationship tracking, and permission-scoped hybrid retrieval (`Precision@k`, `Recall@k`, `MRR`, `nDCG`).
- **Next Expansion:** Full temporal interval algebra (Allen's interval relations) and interactive graph visualization.

## Phase 6 — Imaging and Multimodal Research (**Baseline Guardrails Implemented & Software-tested**)
- **Deliverables:** `CustomMedicalVisionNet-v1` (quality/modality/ROI baseline) and `CustomMultimodalFusionNet-v1` enforcing explicit `verified_pair_id` checks and cross-modal disagreement detection.
- **Next Expansion:** Custom 2D U-Net segmentation on authorized annotated medical imaging datasets.

## Phase 7 — Security, Governance, and Institutional Workflows (**Core Implemented & Software-tested**)
- **Deliverables:** Argon2id authentication, HMAC-SHA256 scoped tokens, fine-grained RBAC/ABAC separating medical data access from training and auditing, cross-institution isolation, bilingual PII scrubbing with interval-preserving date shifting, isolated pseudonym vault, and tamper-evident SHA-256 hash-chained audit logs.
- **Next Expansion:** Hardware MFA/WebAuthn integration and automated retention/expiration purge jobs.

## Phase 8 — Federated Research (**Multi-Node Simulation Implemented & Experimentally validated**)
- **Deliverables:** `FedAvg` and `FedProx` sample-weighted aggregation, L2-norm Byzantine poisoning quarantine, and Differential Privacy Gaussian noise accounting ($\epsilon$).
- **Next Expansion:** Mutual-TLS network transport between authorized hospital nodes (requires explicit institutional agreements and human approval).

## Phase 9 & Phase 10 — Advanced Scientific Discovery & Release Readiness (**Planned**)
- **Deliverables:** Authorized literature monitoring (PubMed/EuropePMC), hypothesis power/sample-size calculators, TRIPOD+AI / CONSORT-AI checklist validators, and long-term archival packaging.
