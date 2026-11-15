# Project Status

**Last Updated:** 2026-11-15  
**Active Phase:** Phase 12 — Complete (`hybrid_v2` CRF+ConText/NegEx ensemble, column-aware OCR layout segmentation, and `realistic_80` EN+FR evaluation; Python 3.13 pure-NumPy runtime confirmed per `ADR-0009`)

Status vocabulary: `Planned` | `In progress` | `Implemented` | `Tested` | `Evaluated on <dataset, n docs>`

---

## 1. Machine & Environment Baseline

- **OS:** macOS Darwin `x86_64`
- **CPU:** Intel(R) Core(TM) i7-9750H @ 2.60GHz (6 cores / 12 threads)
- **RAM:** 32 GB (CPU-only execution)
- **Python:** 3.13.12 | **Node.js:** v24.18.0

---

## 2. Component Status Matrix

| Component | Status | Notes |
|---|---|---|
| Packaging, CI, ADRs 0005–0008, `extras/` isolation | **Implemented / Tested** | `pyproject.toml` fixed, hard-coded password removed, dermatology/vision isolated to `extras/` |
| Document parsers & OCR (TXT, digital/scanned PDF, DOCX, PNG/JPEG/TIFF/WEBP) | **Evaluated on `SYNTH-MED-V3.1` `test_iid` (120 pages = 60 EN + 60 FR)** | Overall CER `0.1927` (`en` `0.1895`, `fr` `0.1959`), WER `0.3198`; end-to-end F1 drop `-0.4322` (`classical_crf`), `-0.4605` (`hybrid`) (`docs/results/v3_1_ocr_full_benchmark.json`) |
| Bilingual terminology registry (EN/FR) | **Tested** | Static registry; corpus-scale normalization evaluation pending |
| Security (Argon2id auth, HMAC tokens, PII scrubber, audit chain) | **Tested** | Hard-coded demo password removed; uses `MEDRESEARCH_DEMO_PASSWORD` env var |
| Output contract (`extraction/contract.py` & `extraction/schema_models.py`) | **Tested** | Pydantic v2 single-source-of-truth + exported `schema/medical_extraction.schema.json` + `/api/v1/schema/extraction` |
| Annotated EN/FR corpus (`>= 300 EN + >= 300 FR`) | **Implemented / Audited (`SYNTH-MED-V3.1`, 1,240 docs)** | `SYNTH-MED-V3.1` (`data/processed/v3_1/`): 620 EN + 620 FR, 6 structural doc types without title-header leakage, `train` (600), `val` (160), `test_iid` (160), `test_unseen_terms` (160, 0% term overlap), `test_unseen_templates` (160, 0.6% masked overlap) |
| Evaluation harness (CI, 10k bootstrap, Holm-Bonferroni, error taxonomy, profiler) | **Tested (`SYNTH-MED-V3.1` 5-seed suite)** | Vectorized 10,000-resample cluster-paired bootstrap + 21-pair Holm-Bonferroni tables per split |
| 5 Single Approaches (Rule, TF-IDF+ML/CRF, BiLSTM, BiLSTM-CRF, Transformer) | **Evaluated on `SYNTH-MED-V3.1` (all 4 splits, seeds 1–5)** | `classical_crf` Strict F1 `0.9968` IID / `0.9169` unseen terms / `0.9527` unseen templates; `classical_ml` `0.9983` / `0.8441` / `0.9670`; `bilstm_crf` `0.9884` / `0.8141` / `0.9392`; stacked BPE `transformer` `0.7930` / `0.5241` / `0.6886` |
| 5-Stage Hybrid Pipeline | **Evaluated on `SYNTH-MED-V3.1` (all 4 splits, seeds 1–5)** | Strict F1 `0.9941` IID / `0.8277` unseen terms / `0.8949` unseen templates; leads Rel F1 (`0.8922` / `0.8127` / `0.8029`) and TIMEX F1 (`0.9986` / `1.0000` / `0.9969`); ablations show Stage 3/4 help, Stage 3b/5 do not |
| FastAPI backend + SQLite jobs & reviews | **Tested** | FastAPI OpenAPI 3.1, `/api/v1/reviews` persistence + `jsonl`/`brat` export, `/react/` SPA fallback |
| React + TypeScript + Tailwind + shadcn-style workspace | **Implemented / Tested (14/14 pages + Playwright E2E)** | OpenAPI-generated types, Dark Mode, Review Queue keyboard workflow, Playwright E2E screenshots (`docs/screenshots/`) |
| EN/FR traceable reports | **Tested** | `medresearch_platform/reports/generator.py` (Markdown, HTML, PDF with negation badges, source `[doc, p., chars]` citations, EN/FR disclaimer), `/api/v1/documents/{id}/report`, `medextract report` |
| Final scientific evaluation & `RESEARCH_REPORT.md` | **Evaluated on `SYNTH-MED-V3.1` (640 eval docs × 5 seeds, `3415.92s`)** | Final 5-seed report published in `docs/RESEARCH_REPORT.md` from `docs/results/v3_1_final_locked_eval.json` |

---

## 3. Questions for the Owner

1. **NumPy vs PyTorch (`DECISIONS/ADR-0009-numpy-vs-pytorch.md`):** Currently the neural models run in pure NumPy because Python 3.13 on macOS `x86_64` has no official `torch` wheel. Installing Python 3.12 (e.g. via `uv python install 3.12` or `pyenv`) enables `torch 2.2.2` CPU on Intel Mac (and newer `torch` on Windows/Linux). Do you want to stay on pure NumPy (zero C++ runtime issues, slower training, smaller models) or switch to Python 3.12 + PyTorch CPU? Keeping NumPy until you decide.
2. **Realistic Free-Text Evaluation Set (Phase 11 Step 2.6):** After Steps 1–2, will you author/annotate 40 EN + 40 FR free-text notes (using our Label Studio protocol) or provide a licensed public corpus?
