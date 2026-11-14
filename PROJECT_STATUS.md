# Project Status

**Last Updated:** 2026-11-14  
**Active Phase:** Phase 11 — correction (Step 6 complete: fresh-clone CI & `test_frontend_assets_and_security_headers` F8 fixed; Step 7 frontend pages/charts/pdf.js/Vitest F7 next)

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
| Document parsers & OCR (TXT, digital/scanned PDF, DOCX, PNG/JPEG/TIFF/WEBP) | **Evaluated on `SYNTH-MED-V3` `val` (16 docs × 3 tiers = 48 scans)** | `clean_scan` CER `0.0021`/WER `0.0138`; `mild_scan` CER `0.0022`/WER `0.0119`; `heavy_scan` CER `0.1615`/WER `0.5471` (`docs/results/ocr_stress_benchmark.json`) |
| Bilingual terminology registry (EN/FR) | **Tested** | Static registry; corpus-scale normalization evaluation pending |
| Security (Argon2id auth, HMAC tokens, PII scrubber, audit chain) | **Tested** | Hard-coded demo password removed; uses `MEDRESEARCH_DEMO_PASSWORD` env var |
| Output contract (`extraction/contract.py` & `extraction/schema_models.py`) | **Tested** | Pydantic v2 single-source-of-truth + exported `schema/medical_extraction.schema.json` + `/api/v1/schema/extraction` |
| Annotated EN/FR corpus (`>= 300 EN + >= 300 FR`) | **Implemented / Audited (`SYNTH-MED-V3`, 1,240 docs)** | `SYNTH-MED-V3` built with 640+ templates, expanded EN/FR lexicon, `train` (600), `val` (160), `test_iid` (160), `test_unseen_terms` (160, 0% clinical term overlap), `test_unseen_templates` (160, 0.6% masked sentence overlap); `0` doubled docs |
| Evaluation harness (CI, McNemar, error taxonomy, profiler) | **Tested (`SYNTH-MED-V3` multi-split ready)** | Supports `val`, `test`, `test_iid`, `test_unseen_terms`, `test_unseen_templates` |
| 5 Single Approaches (Rule, TF-IDF+ML/CRF, BiLSTM, BiLSTM-CRF, Transformer) | **Tested on `SYNTH-MED-V3` `val` (`bilstm_crf`: `0.9604 ± 0.0017`)** | 32-dim char prefix/suffix morphology + seeded word/feature dropout (`F3` fixed); full 5-approach benchmark in Step 8 |
| 5-Stage Hybrid Pipeline | **Tested on `SYNTH-MED-V3` (F4/F5 fixed)** | Stage 3 balanced assertion BiLSTM+ConText fusion & type-masked CNN relations; Stage 5 text-reading Transformer reranker (`99`-dim span+left/right ctx+char+meta); full benchmark in Step 8 |
| FastAPI backend + SQLite jobs | **Tested** | Migrated from Starlette to FastAPI (`OpenAPI 3.1`), multipart `/api/v1/documents/upload`, SQLite `jobs` table + SSE `/api/v1/jobs/{id}/events` |
| React + TypeScript + Tailwind + shadcn-style workspace | **In progress (5/14 pages)** | Phase 7 partial (login, workspace, upload, benchmark, timeline); remaining pages, pdf.js, charts, Vitest & Playwright in Phase 11 Step 7 |
| EN/FR traceable reports | **Tested** | `medresearch_platform/reports/generator.py` (Markdown, HTML, PDF with negation badges, source `[doc, p., chars]` citations, EN/FR disclaimer), `/api/v1/documents/{id}/report`, `medextract report` |
| Final scientific evaluation & `RESEARCH_REPORT.md` | **Invalidated (F1/F2)** | Old report superseded & archived in `docs/results/archive/2026-11-14_synth_v2_buggy/`; new `SYNTH-MED-V3` report in Phase 11 Step 9 |

---

## 3. Questions for the Owner

1. **NumPy vs PyTorch (`DECISIONS/ADR-0009-numpy-vs-pytorch.md`):** Currently the neural models run in pure NumPy because Python 3.13 on macOS `x86_64` has no official `torch` wheel. Installing Python 3.12 (e.g. via `uv python install 3.12` or `pyenv`) enables `torch 2.2.2` CPU on Intel Mac (and newer `torch` on Windows/Linux). Do you want to stay on pure NumPy (zero C++ runtime issues, slower training, smaller models) or switch to Python 3.12 + PyTorch CPU? Keeping NumPy until you decide.
2. **Realistic Free-Text Evaluation Set (Phase 11 Step 2.6):** After Steps 1–2, will you author/annotate 40 EN + 40 FR free-text notes (using our Label Studio protocol) or provide a licensed public corpus?
