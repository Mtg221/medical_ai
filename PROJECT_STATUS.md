# Project Status

**Last Updated:** 2026-11-24  
**Maturity:** **Research prototype — evaluated on locked synthetic splits (`SYNTH-MED-V3.1`), `4,882` real open-license clinical & public-domain biomedical documents (`MACCROBAT2020`, `PxCorpus`, `E3C`, `BC5CDR`, `NCBI Disease`), an LLM-authored 80-note free-text proxy (`realistic_llm_80`), external document/skin image datasets (`locked_test_v3`, `EkaCare`, `CORD`), and a blind-agent PHI stress test; human owner-written & double-annotated free-text + PHI evaluation pending owner notes** (see [`docs/REALISTIC_SET_PROTOCOL.md`](docs/REALISTIC_SET_PROTOCOL.md) and [`docs/REALISTIC_SET_OWNER_CHECKLIST.md`](docs/REALISTIC_SET_OWNER_CHECKLIST.md)).

Status vocabulary: `Planned` | `In progress` | `Implemented` | `Tested` | `Evaluated on <dataset, n docs>`

---

## 1. Machine & Environment Baseline

- **OS:** macOS Darwin `x86_64` (CI matrix: `ubuntu-latest`, `windows-latest` × Python `3.12`/`3.13`)
- **CPU:** Intel(R) Core(TM) i7-9750H @ 2.60GHz (6 cores / 12 threads)
- **RAM:** 32 GB (CPU-only execution)
- **Python:** 3.12 (`.venv-vision`) & 3.13 (base) | **Node.js:** v22/v24

---

## 2. Component Status Matrix

| Component | Status | Notes |
|---|---|---|
| Packaging, CI, ADRs 0001–0010, `extras/` isolation | **Implemented / Tested** | Version aligned to `0.2.0`; Intel macOS `.[vision]` `numpy<2` compatibility constraint in `pyproject.toml`; fresh-clone `pytest` CI job; `file:///` doc link guard; `extras/legacy/` isolated per `ADR-0008` |
| Document parsers & OCR (TXT, digital/scanned PDF, DOCX, PNG/JPEG/TIFF/WEBP) | **Evaluated on `SYNTH-MED-V3.1` `test_iid` (120 pages = 60 EN + 60 FR)** | Character-level `map_ocr_spans_to_reference` + `detect_column_split`: overall CER `0.0862` (`en` `0.0824`, `fr` `0.0899`), WER `0.2112`; `classical_crf` Strict F1 `0.9978 -> 0.9128` (`-0.0850`), `hybrid` `0.9937 -> 0.8799` (`-0.1139`) (`docs/results/v3_1_ocr_full_benchmark.json`) |
| Security & bilingual PII redaction (`privacy.py`, `person_age_detector.py`, `auth.py`, `audit.py`) | **Hardened & Audited (Phases 1–3 complete); human PHI set pending** | Argon2id auth, HMAC tokens with `jti` revocation (`/api/v1/auth/logout`), sliding-window login brute-force throttling (`429`), zero `joblib`/pickle deserialization (safe JSON+NPZ + `torch.load(..., weights_only=True)`), SHA-256 hash-chained audit logs, eponym-protected person/age pre-filter. Blind-agent 60-sentence stress test exposed `0.8304` substring / `0.6696` strict-token recall (`docs/results/p10_redaction_blind_agent_eval.json`); post-hardening `1.0000` (`post-hoc`, `docs/results/p10_redaction_post_fix_eval.json`). Owner runner ready: `scripts/eval_redaction_human.py` + `docs/PHI_REDACTION_OWNER_TEMPLATE.jsonl` |
| Output contract (`extraction/contract.py` & `extraction/schema_models.py`) | **Tested** | Pydantic v2 single-source-of-truth + exported `schema/medical_extraction.schema.json` + `/api/v1/schema/extraction` |
| Annotated EN/FR synthetic corpus (`SYNTH-MED-V3.1`, 1,240 docs) | **Locked & Verified (`data/locked/synth_med_v3_1/LOCK_MANIFEST.json`)** | Restored `5f5e3fe` SHA-256 locked splits (`train` 600, `val` 160, `test_iid` 160, `test_unseen_terms` 160, `test_unseen_templates` 160); 6 structural doc types without header leakage; CRLF-normalized hashing on Windows |
| Free-text evaluation & double-annotation IAA tooling | **Evaluated on `realistic_llm_80` (proxy); human owner set pending** | LLM-authored proxy (`owner_written: false`, `docs/results/realistic_llm_80_eval.json`): `hybrid` `0.6640` Strict F1 (`0.7372` Lenient), `bilstm_crf` `0.6307`, `classical_crf` `0.6271`. Upgraded `configs/label_studio_medical_en_fr.xml`, `scripts/import_label_studio_realistic.py`, `scripts/eval_realistic.py`, and `medresearch_platform/eval_harness/iaa.py` now support entities, assertions, relations, TIMEX, and double-annotation IAA (Cohen's kappa + span/relation/TIMEX F1) |
| Evaluation harness (5-seed, 10k bootstrap, Holm-Bonferroni, IAA, error taxonomy, profiler) | **Tested (`SYNTH-MED-V3.1` 5-seed suite)** | 10,000-resample 160-document bootstrap 95% CIs + separate seed-std + 21-pair Holm-Bonferroni tables + `TEST_SET_LOG.md` post-hoc discipline |
| 6 Single Approaches (Rule, TF-IDF+SVM, CRF, BiLSTM, BiLSTM-CRF, Transformer) | **Evaluated on `SYNTH-MED-V3.1` (all 4 splits, seeds 1–5) + `realistic_llm_80`** | All 6 neural models pass `< 1e-4` finite-difference gradient checks (`tests/test_evaluated_neural_gradients.py`) |
| 5-Stage Hybrid & Hospital 18-Model Cross-Architecture (BiLSTM + Medical Transformer) Ensemble (`torch_ner.py`, `pipeline.py`, `stage4_temporal.py`) | **Evaluated, Served & Hardened (`models/hospital_char_cnn_bilstm_crf_seed1..18/`)** | `hybrid_v2` (`0.9972` `test_iid`, `0.9204` `test_unseen_terms`, `0.9310` `test_unseen_templates`); Stage 4 `normalize_timex` (`39/40 = 0.9750`); **18-Model Cross-Architecture Hospital NLP Ensemble** (`15 BiLSTM + 3 From-Scratch 4-Layer Medical Transformer pretrained with MLM on 6,518 E3C Layer 3 EN/FR hospital notes`, `12/18` supermajority + CRF v2, and `confidence_variance` / `high_uncertainty` guardrails): `open_clinical_test` **`0.6779` Strict / `0.7484` Lenient (Precision `0.7226`)**, `realistic_llm_80` `0.7197` Strict (P `0.7683`), `test_unseen_terms` `0.9342` (P `0.9572`) (`docs/results/hospital_18model_cross_arch_ensemble_benchmark.json`) |
| Dermatology skin-image vision track (`medresearch_platform/vision/`) | **Evaluated on `locked_test_v3` (`n=1,624`) + `locked_test_v4` (`n=1,611`, 0 exact SHA-256 dups) + EkaCare (`125`) + CORD (`100`)** | Lesion-grouped `v3` splits (`0` cross-split group overlap) + published `v3_leakage_audit.json` & SHA-256-deduplicated `locked_test_v4` (`v4_dedup_audit.json`, `0` exact byte duplicates). Lazy on-disk PyTorch `Dataset` for `>=320x320` training (`train_vision_gpu.py`), zero-leakage external smartphone ingestion (`scripts/ingest_external_clinical_derm.py` for PAD-UFES-20 / Fitzpatrick17k / DDI), and prospective shadow-mode gate harness (`docs/SHADOW_MODE_CLINICAL_PROTOCOL.md`, `scripts/eval_shadow_mode.py`). **9-Seed Cross-Architecture Modality-Aware Vision Ensemble (`7x SE-ResNetV2 + 2x From-Scratch ConvNeXt-Derm`, `4x TTA`, Fitzpatrick-inclusive Lesion ROI Multi-Crop & bilingual retake guidance)**: Overall AUROC **`0.9353`**, **NEW RECORD `Sens@90%Spec` `0.7952`**, Brier **`0.1009`**, ECE **`0.0315`** (Dermoscopic `0.9734` / `93.33%` sens@90, Clinical Smartphone `0.8784` with ROI multi-crop / `58.79%` sens@90, TBP `0.8763` auto-abstained); **Uncertainty Guardrail (`std > 0.12`)**: low-uncertainty **`0.9419` AUROC / `12.9%` err (`n=1,528`)** (`docs/results/vision/vision_9seed_cross_arch_benchmark.json`). Reproducible artifact manifest (`30/30` verified): `docs/results/ARTIFACTS_MANIFEST.json`. Executive deck: `docs/presentation/MINISTER_HOSPITAL_PRESENTATION.html` |
| FastAPI backend + SQLite jobs & reviews | **Refactored & Tested (`interfaces/routers/`)** | Modularized `api.py` (1,135 lines → 121-line factory + 6 domain router modules under `medresearch_platform/interfaces/routers/`), OpenAPI 3.1, `401`/`403` Bearer middleware, `415`/`413` upload guards, SSE job streaming, `/api/v1/reviews` persistence + `jsonl`/`brat` export |
| React + TypeScript + Tailwind workspace (`frontend/`) | **Implemented / Tested (16 pages + Vitest + E2E + Lighthouse 100)** | Bilingual EN/FR UI, shared `entityStyles.ts`, Relation Graph, Vision & Benchmark pages, plain EN/FR served capability notices, Lighthouse `100/100` accessibility |
| EN/FR traceable reports & `RESEARCH_REPORT.md` | **Tested & Cleaned** | `medresearch_platform/reports/generator.py` (Markdown, HTML, PDF with negation badges, source citations, EN/FR clinical disclaimers); `docs/RESEARCH_REPORT.md` Section 15 formatting defect fixed |

---

## 3. Pending Human Owner Inputs

1. **Human Owner-Written & Double-Annotated Free-Text Notes (`docs/REALISTIC_SET_OWNER_CHECKLIST.md`):** 40 EN + 40 FR notes covering entities, assertions, relations, and TIMEX with `>= 20%` double annotation in Label Studio (`configs/label_studio_medical_en_fr.xml`), imported with `--iaa-out` via `scripts/import_label_studio_realistic.py`.
2. **Unseen Human PHI Redaction Set (`docs/PHI_REDACTION_OWNER_TEMPLATE.jsonl`):** `>= 50` unseen EN/FR PHI sentences + `>= 10` tricky non-PHI sentences evaluated once with `scripts/eval_redaction_human.py --owner-written`.
3. **Owner-Taken Smartphone Document Photos:** Optional final router confirmation on real owner-captured phone photos of printed clinical notes/lab sheets.

---

## 4. Deep Source-Code Audit Remediation Status (`F-01`–`F-11`)

| Finding | Severity | Status | Evidence / Resolution |
|---|---|---|---|
| `F-01` Reproducible artifact packaging | Critical | **Resolved (Local Bundle Ready)** | `scripts/package_release_artifacts.py` (`verify`/`pack`/`unpack`), `docs/results/ARTIFACTS_MANIFEST.json` (`30/30` verified), `dist/release/*.tar.gz` + `checksums.sha256` |
| `F-02` Safe model deserialization | High | **Resolved & Tested** | `weights_only=True` on all `torch.load`; zero `joblib.load`/`pickle.load` (safe JSON+NPZ in `persistence.py`); guarded by `test_no_unsafe_deserialization_calls_in_platform` |
| `F-03` Locked-test reuse & deduplication | High | **Mitigated (`v4` Split Built)** | `configs/vision/locked_test_v4.txt` (`n=1,611`, `0` exact SHA-256 duplicates), `vision_9seed_locked_test_v4_benchmark.json` (`0.9348` AUROC / `0.7962` Sens@90% Spec); blind human owner sets pending |
| `F-04` Clinical smartphone & shadow-mode readiness | High | **Engineering Complete** | `docs/SHADOW_MODE_CLINICAL_PROTOCOL.md`, `scripts/eval_shadow_mode.py` (8 promotion gates incl. Fitzpatrick parity), `scripts/ingest_external_clinical_derm.py` (PAD-UFES-20 / Fitzpatrick17k / DDI) |
| `F-05` Login brute-force & token revocation | Medium | **Resolved & Tested** | Sliding-window `LoginRateLimiter` (`429`) + `jti` `revoked_tokens` table + `POST /api/v1/auth/logout` (`test_login_rate_limiting_and_token_revocation`) |
| `F-06` Temporal calendar validation | Medium | **Resolved & Tested** | Strict `datetime.date(y, m, d)` validation rejecting impossible calendar dates in `stage4_temporal.py` (`test_pod_and_invalid_calendar_dates`) |
| `F-07` Docs sync & root `LICENSE` | Medium | **Resolved** | Added root MIT `LICENSE` linked in `pyproject.toml`; synced `README.md` Section 1.2 with `PROJECT_STATUS.md` |
| `F-08` Local checkpoint integrity | Medium | **Verified Clean** | `30/30` served checkpoints pass SHA-256 verification (`0` invalid, `0` pretrained) |
| `F-09` Vision GPU host RAM OOM at `320px` | Medium | **Resolved & Tested** | Lazy on-disk `_LazyImageDataset` in `cnn_torch.py` + `train_vision_gpu.py` (`test_lazy_on_disk_vision_dataset_fit_and_predict`) |
| `F-10` Vision `v3` visual leakage audit | Low | **Resolved & Published** | `scripts/audit_v3_leakage.py` → `docs/results/vision/v3_leakage_audit.json` & `v4_dedup_audit.json` |
| `F-11` Legacy UI header & stale lockfile | Low | **Resolved** | `X-Workspace-UI: legacy-fallback; canonical=/react/` header on `/` (preserving zero-build fallback) + regenerated `requirements-lock.txt` |
