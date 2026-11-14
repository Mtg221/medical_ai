# Medical Information Intelligence & Federated Scientific Research Platform

**Version:** `0.1.0`  
**Specification Coverage:** `v23.2` through `v40.0` (`MASTER_SPECIFICATION_AND_IMPLEMENTATION_PROMPT.md`)  
**Primary Languages:** English (`en`) & French (`fr`) + Mixed (`mixed_en_fr`)  
**Execution Profile:** CPU-Only Native (`Intel Core i7-9750H`, 32 GB RAM, macOS Darwin `x86_64`)  
**Primary AI Track:** Custom architectures initialized strictly from random weights (zero pretrained weights)

> **Clinical & Scientific Safety Notice**  
> This platform is a **research and medical information extraction support system**. It is **NOT** intended to diagnose patients autonomously, prescribe/modify/discontinue treatment, replace clinicians, or present AI-extracted/inferred outputs as verified clinical truth. All experimental validations to date are on the synthetic bilingual corpus `SYNTH-MED-EN-FR v1.0.0` (`Experimentally validated` on synthetic benchmark; **NOT** `Clinically validated`).

---

## 1. Key Implemented Capabilities

- **Six Independent AI Approaches (Random-Init Primary Track):**
  1. **Medical Document / OCR Intelligence (`CustomOCRConvNet-v1`):** Native PDF, DOCX, TXT, and PNG parsing, clinical section segmentation, and custom 2D Patch-Conv + MLP OCR with per-block/per-char confidence, aleatoric entropy, epistemic margin uncertainty, and CER/WER benchmarking.
  2. **Medical NLP (`CustomMedicalNLPNet-v1`):** Multi-head neural network extracting 12 clinical entity types, 6 assertion states (`AFFIRMED`, `NEGATED`, `UNCERTAIN`, `HYPOTHETICAL`, `CONDITIONAL`, `FAMILY_HISTORY`), 5 temporal states, medication states, and same-clause clinical relations.
  3. **Medical Computer Vision (`CustomMedicalVisionNet-v1`):** Image integrity/contrast quality assessment, modality verification, and foreground ROI bounding box estimation.
  4. **Multimodal Intelligence (`CustomMultimodalFusionNet-v1`):** Cross-modal disagreement analysis strictly gated by explicit `verified_pair_id` (prevents folder co-location pairing assumptions).
  5. **Longitudinal & Knowledge Graph Intelligence:** Subject timeline reconstruction, cross-document contradiction detection, and explicit `EXTRACTED` vs `INFERRED` tracking.
  6. **Hybrid Retrieval (`CustomHybridRetrievalNet-v1`):** Permission-scoped lexical, structured terminology, and custom random-init neural similarity search (`Precision@k`, `Recall@k`, `MRR`, `nDCG`).
- **Bilingual Terminology Registry:** Versioned EN/FR concepts mapped to `ICD-10`, `SNOMED-CT`, `LOINC`, `RxNorm`, and `ATC`, preserving `EXACT`, `SYNONYM`, `APPROXIMATE`, `AMBIGUOUS`, and `UNMAPPED` states.
- **Zero-Trust Security & Privacy:** Argon2id password hashing, HMAC-SHA256 scoped tokens, RBAC/ABAC separating medical data access from training and auditing, institution isolation, bilingual PII redaction with interval-preserving date shifting, isolated pseudonym vault, DOCX zip-bomb & prompt-injection guards, and tamper-evident SHA-256 hash-chained audit logs.
- **Federated Simulation & Evaluation:** `FedAvg` / `FedProx` aggregation, L2-norm Byzantine outlier quarantine, Differential Privacy $\epsilon$ accounting, ECE, Brier score, selective prediction coverage-risk curves, bootstrap 95% CIs, and EN/FR subgroup fairness.

---

## 2. Quick Start & Verification Commands

```bash
# 1. Inspect hardware, OS, and dependency readiness
python3 -m medresearch_platform.interfaces.cli diagnostics

# 2. Initialize database schema, terminology, and default institutions/roles
python3 -m medresearch_platform.interfaces.cli init-db

# 3. Seed synthetic bilingual corpus (SYNTH-MED-EN-FR v1.0.0) with leak-free split verification
python3 -m medresearch_platform.interfaces.cli seed

# 4. Run full CPU-only Phase 1 OCR, NLP, Retrieval, Fairness & Federated benchmark
python3 -m medresearch_platform.interfaces.cli run-experiment

# 5. Verify tamper-evident SHA-256 audit log chain
python3 -m medresearch_platform.interfaces.cli verify-audit

# 6. Run full automated unit, security, scientific, and API/CLI integration test suite
python3 -m pytest -v
```

---

## 3. REST API & Accessible Web Workspace

The Starlette ASGI application (`medresearch_platform.interfaces.api:create_app`) provides:
- `GET /` — Accessible (WCAG 2.2 AA target) bilingual EN/FR research workspace with epistemic provenance legend.
- `GET /api/v1/health` — Hardware, OS, and readiness diagnostics.
- `GET /api/v1/openapi.json` — OpenAPI 3.1 contract.
- `POST /api/v1/auth/token` — Authenticate and issue scoped HMAC Bearer token.
- `GET /api/v1/documents` & `GET /api/v1/documents/{document_id}` — Institution-isolated document, OCR, entity, and relation inspection.
- `POST /api/v1/documents/ingest` — Validate, store immutably, de-identify, parse, and extract a document.
- `POST /api/v1/entities/{entity_id}/review` — Submit human expert review/correction without erasing AI prediction history.
- `GET /api/v1/subjects/{subject_id}/timeline` — Chronological timeline and cross-document contradiction flags.
- `GET /api/v1/search?q=...` — Permission-scoped hybrid retrieval.
- `GET /api/v1/terminology/map?term=...` — Bilingual terminology normalization.
- `GET /api/v1/models` & `GET /api/v1/experiments` & `POST /api/v1/experiments/run` — Model registry and reproducible experiment manifests.
- `GET /api/v1/audit/verify` — Cryptographic audit chain verification.

---

## 4. Documentation Index

- [ARCHITECTURE.md](file:///Users/guest777/Desktop/medical/ARCHITECTURE.md) — Modular monolith architecture diagram and domain boundaries.
- [PROJECT_STATUS.md](file:///Users/guest777/Desktop/medical/PROJECT_STATUS.md) — Verified baseline and requirement traceability matrix (using Section 47 status vocabulary).
- [ROADMAP.md](file:///Users/guest777/Desktop/medical/ROADMAP.md) — Phased implementation plan (Phases 0–10).
- [CHANGELOG.md](file:///Users/guest777/Desktop/medical/CHANGELOG.md) — Versioned release history.
- [DECISIONS/](file:///Users/guest777/Desktop/medical/DECISIONS) — Architecture Decision Records (`ADR-0001` to `ADR-0004`).
- [docs/data-governance/DATA_POLICY_AND_SYNTHETIC_CORPUS.md](file:///Users/guest777/Desktop/medical/docs/data-governance/DATA_POLICY_AND_SYNTHETIC_CORPUS.md) — Synthetic data policy and leak-free split verification.
- [docs/model-cards/](file:///Users/guest777/Desktop/medical/docs/model-cards) — Model cards for `CustomOCRConvNet-v1` and `CustomMedicalNLPNet-v1`.
- [docs/experiments/PHASE1_BASELINE_EXPERIMENTS.md](file:///Users/guest777/Desktop/medical/docs/experiments/PHASE1_BASELINE_EXPERIMENTS.md) — Phase 1 CPU-only experiment results and resource measurements.
- [docs/security/SECURITY_AND_PRIVACY_REVIEW.md](file:///Users/guest777/Desktop/medical/docs/security/SECURITY_AND_PRIVACY_REVIEW.md) — Threat model, RBAC/ABAC, PII scrubbing, and audit chain.
- [docs/reproducibility/REPRODUCIBILITY_GUIDE.md](file:///Users/guest777/Desktop/medical/docs/reproducibility/REPRODUCIBILITY_GUIDE.md) — Random-init provenance and deterministic experiment reproduction.
- [docs/installation/INSTALLATION_AND_DIAGNOSTICS.md](file:///Users/guest777/Desktop/medical/docs/installation/INSTALLATION_AND_DIAGNOSTICS.md) — Native CPU-only setup, PostgreSQL 18 / SQLite portability, and troubleshooting.
