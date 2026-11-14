# Project Status

**Last Updated:** 2026-11-14  
**Active Phase:** Phase 11 — correction (Step 0 complete: audit reproduced F1/F2, v2 results archived)

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
| Document parsers & OCR (TXT, digital/scanned PDF, DOCX, PNG/JPEG/TIFF/WEBP) | **Tested (EN/FR scan benchmark: CER 0.0, WER 0.0)** | Page-by-page `pdfplumber` word boxes & lab tables, `pypdfium2` 300 dpi render, OpenCV preprocessing (upscale, deskew, denoise, adaptive threshold), Tesseract 5.5.4 (`eng+fra`), mixed-page `HYBRID` mode |
| Bilingual terminology registry (EN/FR) | **Tested** | Static registry; corpus-scale normalization evaluation pending |
| Security (Argon2id auth, HMAC tokens, PII scrubber, audit chain) | **Tested** | Hard-coded demo password removed; uses `MEDRESEARCH_DEMO_PASSWORD` env var |
| Output contract (`extraction/contract.py` & `extraction/schema_models.py`) | **Tested** | Pydantic v2 single-source-of-truth + exported `schema/medical_extraction.schema.json` + `/api/v1/schema/extraction` |
| Annotated EN/FR corpus (`>= 300 EN + >= 300 FR`) | **Invalidated (F1/F2)** (`SYNTH-MED-V2`) | `SYNTH-MED-V2` (800 docs) had doubled TIMEX text in 100% of docs and `94.8%` masked sentence overlap; `SYNTH-MED-V3` planned in Phase 11 Steps 1–2 |
| Evaluation harness (CI, McNemar, error taxonomy, profiler) | **Tested** | Hand-made unit tests pass; seed/CI/multi-test-set upgrades in Phase 11 Step 3 |
| 5 Single Approaches (Rule, TF-IDF+ML/CRF, BiLSTM, BiLSTM-CRF, Transformer) | **Invalidated (F1/F2)** | Previous numbers measured memorisation on `SYNTH-MED-V2`; char-level + generalization fixes in Phase 11 Steps 3–4 |
| 5-Stage Hybrid Pipeline | **Invalidated (F1/F2, F4/F5)** | Previous numbers invalid; Stage 3 hurt assertion/relation scores in ablation and Stage 5 lacked token-level reading (Phase 11 Step 5) |
| FastAPI backend + SQLite jobs | **Tested** | Migrated from Starlette to FastAPI (`OpenAPI 3.1`), multipart `/api/v1/documents/upload`, SQLite `jobs` table + SSE `/api/v1/jobs/{id}/events` |
| React + TypeScript + Tailwind + shadcn-style workspace | **In progress (5/14 pages)** | Phase 7 partial (login, workspace, upload, benchmark, timeline); remaining pages, pdf.js, charts, Vitest & Playwright in Phase 11 Step 7 |
| EN/FR traceable reports | **Tested** | `medresearch_platform/reports/generator.py` (Markdown, HTML, PDF with negation badges, source `[doc, p., chars]` citations, EN/FR disclaimer), `/api/v1/documents/{id}/report`, `medextract report` |
| Final scientific evaluation & `RESEARCH_REPORT.md` | **Invalidated (F1/F2)** | Old report superseded & archived in `docs/results/archive/2026-11-14_synth_v2_buggy/`; new `SYNTH-MED-V3` report in Phase 11 Step 9 |

---

## 3. Questions for the Owner

1. **NumPy vs PyTorch (`DECISIONS/ADR-0009-numpy-vs-pytorch.md`):** Currently the neural models run in pure NumPy because Python 3.13 on macOS `x86_64` has no official `torch` wheel. Installing Python 3.12 (e.g. via `uv python install 3.12` or `pyenv`) enables `torch 2.2.2` CPU on Intel Mac (and newer `torch` on Windows/Linux). Do you want to stay on pure NumPy (zero C++ runtime issues, slower training, smaller models) or switch to Python 3.12 + PyTorch CPU? Keeping NumPy until you decide.
2. **Realistic Free-Text Evaluation Set (Phase 11 Step 2.6):** After Steps 1–2, will you author/annotate 40 EN + 40 FR free-text notes (using our Label Studio protocol) or provide a licensed public corpus?
