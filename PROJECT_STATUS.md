# Project Status

**Last Updated:** 2026-11-19  
**Maturity:** **Research prototype — evaluated on locked synthetic splits (`SYNTH-MED-V3.1`), an LLM-authored 80-note free-text proxy (`realistic_llm_80`), external document/skin image datasets, and a blind-agent PHI stress test; human owner-written & double-annotated free-text + PHI evaluation pending owner notes** (see [`docs/REALISTIC_SET_PROTOCOL.md`](docs/REALISTIC_SET_PROTOCOL.md) and [`docs/REALISTIC_SET_OWNER_CHECKLIST.md`](docs/REALISTIC_SET_OWNER_CHECKLIST.md)).

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
| Document parsers & OCR (TXT, digital/scanned PDF, DOCX, PNG/JPEG/TIFF/WEBP) | **Evaluated on `SYNTH-MED-V3.1` `test_iid` (120 pages = 60 EN + 60 FR)** | Character-level `map_ocr_spans_to_reference` + `detect_column_split`: overall CER `0.0862` (`en` `0.0725`, `fr` `0.0999`), WER `0.2112`; `classical_crf` Strict F1 `0.9978 -> 0.9128` (`-0.0850`), `hybrid` `0.9937 -> 0.8799` (`-0.1139`) (`docs/results/v3_1_ocr_full_benchmark.json`) |
| Security & bilingual PII redaction (`privacy.py`, `person_age_detector.py`, `auth.py`, `audit.py`) | **Evaluated on dev (52) & blind-agent (60) sets; human set pending** | Argon2id auth, HMAC tokens, SHA-256 hash-chained audit logs, eponym-protected person/age pre-filter. Blind-agent 60-sentence stress test exposed `0.8304` substring / `0.6696` strict-token recall (`docs/results/p10_redaction_blind_agent_eval.json`); post-hardening `1.0000` (`post-hoc`, `docs/results/p10_redaction_post_fix_eval.json`). Owner runner ready: `scripts/eval_redaction_human.py` + `docs/PHI_REDACTION_OWNER_TEMPLATE.jsonl` |
| Output contract (`extraction/contract.py` & `extraction/schema_models.py`) | **Tested** | Pydantic v2 single-source-of-truth + exported `schema/medical_extraction.schema.json` + `/api/v1/schema/extraction` |
| Annotated EN/FR synthetic corpus (`SYNTH-MED-V3.1`, 1,240 docs) | **Locked & Verified (`data/locked/synth_med_v3_1/LOCK_MANIFEST.json`)** | Restored `5f5e3fe` SHA-256 locked splits (`train` 600, `val` 160, `test_iid` 160, `test_unseen_terms` 160, `test_unseen_templates` 160); 6 structural doc types without header leakage; CRLF-normalized hashing on Windows |
| Free-text evaluation & double-annotation IAA tooling | **Evaluated on `realistic_llm_80` (proxy); human owner set pending** | LLM-authored proxy (`owner_written: false`, `docs/results/realistic_llm_80_eval.json`): `hybrid` `0.6640` Strict F1 (`0.7372` Lenient), `bilstm_crf` `0.6307`, `classical_crf` `0.6271`. Upgraded `configs/label_studio_medical_en_fr.xml`, `scripts/import_label_studio_realistic.py`, `scripts/eval_realistic.py`, and `medresearch_platform/eval_harness/iaa.py` now support entities, assertions, relations, TIMEX, and double-annotation IAA (Cohen's kappa + span/relation/TIMEX F1) |
| Evaluation harness (5-seed, 10k bootstrap, Holm-Bonferroni, IAA, error taxonomy, profiler) | **Tested (`SYNTH-MED-V3.1` 5-seed suite)** | 10,000-resample 160-document bootstrap 95% CIs + separate seed-std + 21-pair Holm-Bonferroni tables + `TEST_SET_LOG.md` post-hoc discipline |
| 6 Single Approaches (Rule, TF-IDF+SVM, CRF, BiLSTM, BiLSTM-CRF, Transformer) | **Evaluated on `SYNTH-MED-V3.1` (all 4 splits, seeds 1–5) + `realistic_llm_80`** | All 6 neural models pass `< 1e-4` finite-difference gradient checks (`tests/test_evaluated_neural_gradients.py`) |
| 5-Stage Hybrid (`hybrid`, `hybrid_v2`, OOF Stage 5, general EN/FR TIMEX) | **Evaluated, Served & Hardened (`models/hybrid_v2_seed1/`)** | `hybrid_v2` (`val`-selected, `0.9972` `test_iid`, `0.9204` `test_unseen_terms`, `0.9310` `test_unseen_templates`); Stage 4 `normalize_timex` upgraded to general EN/FR parsing (`39/40 = 0.9750` on `tests/test_timex_out_of_format.py`, up from `4/40`); Stage 5 upgraded to out-of-fold (OOF) stacking with class-balanced BCE and calibrated threshold persistence |
| Dermatology skin-image vision track (`medresearch_platform/vision/`) | **Evaluated on `locked_test_v3` (`n=1,624`) + EkaCare (`125`) + CORD (`100`)** | Lesion-grouped `v3` splits (`0` cross-split group overlap). Served API gives risk bands for dermoscopic photos only (AUROC `0.8849`); clinical photos suppress risk bands (`0.7195`), TBP tiles abstain (`95.7%`, `0.4819`), corpus-prevalence served abstention `80.45%` (ECE `0.4335`). Router leaks `0/125` on EkaCare clinical docs and `3/100` non-abstained on CORD smartphone receipts |
| FastAPI backend + SQLite jobs & reviews | **Refactored & Tested (`interfaces/routers/`)** | Modularized `api.py` (1,135 lines → 121-line factory + 6 domain router modules under `medresearch_platform/interfaces/routers/`), OpenAPI 3.1, `401`/`403` Bearer middleware, `415`/`413` upload guards, SSE job streaming, `/api/v1/reviews` persistence + `jsonl`/`brat` export |
| React + TypeScript + Tailwind workspace (`frontend/`) | **Implemented / Tested (16 pages + Vitest + E2E + Lighthouse 100)** | Bilingual EN/FR UI, shared `entityStyles.ts`, Relation Graph, Vision & Benchmark pages, plain EN/FR served capability notices, Lighthouse `100/100` accessibility |
| EN/FR traceable reports & `RESEARCH_REPORT.md` | **Tested & Cleaned** | `medresearch_platform/reports/generator.py` (Markdown, HTML, PDF with negation badges, source citations, EN/FR clinical disclaimers); `docs/RESEARCH_REPORT.md` Section 15 formatting defect fixed |

---

## 3. Pending Human Owner Inputs

1. **Human Owner-Written & Double-Annotated Free-Text Notes (`docs/REALISTIC_SET_OWNER_CHECKLIST.md`):** 40 EN + 40 FR notes covering entities, assertions, relations, and TIMEX with `>= 20%` double annotation in Label Studio (`configs/label_studio_medical_en_fr.xml`), imported with `--iaa-out` via `scripts/import_label_studio_realistic.py`.
2. **Unseen Human PHI Redaction Set (`docs/PHI_REDACTION_OWNER_TEMPLATE.jsonl`):** `>= 50` unseen EN/FR PHI sentences + `>= 10` tricky non-PHI sentences evaluated once with `scripts/eval_redaction_human.py --owner-written`.
3. **Owner-Taken Smartphone Document Photos:** Optional final router confirmation on real owner-captured phone photos of printed clinical notes/lab sheets.
