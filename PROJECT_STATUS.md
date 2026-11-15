# Project Status

**Last Updated:** 2026-11-15  
**Active Phase:** Phase 11B — Step 6 complete (120-page EN+FR OCR benchmark across 6 tiers / 11 degradation modes + end-to-end `classical_crf` & `hybrid` F1 drop + `pytest.skip` without Tesseract); Step 7 (Frontend Phase 7 standard G10) next

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
| 5 Single Approaches (Rule, TF-IDF+ML/CRF, BiLSTM, BiLSTM-CRF, Transformer) | **Evaluated on `SYNTH-MED-V3` (`val` + 3 locked test splits, seeds 1–3)** | `classical_crf` leads NER (`0.9980` IID, `0.9131` unseen terms, `0.9549` unseen templates); `bilstm_crf` (`0.9690` / `0.7860` / `0.9216`); non-zero cross-seed std (`F3` fixed) |
| 5-Stage Hybrid Pipeline | **Evaluated on `SYNTH-MED-V3` (`val` + 3 locked test splits, seeds 1–3)** | `hybrid` (`0.9819` IID, `0.8033` unseen terms, `0.8800` unseen templates; RelF1 `0.9037` / `0.8058` / `0.8264`); `val` ablations confirm Stage 3 Assertion (`+0.0649`) & Relation CNN (`+0.1994`) help |
| FastAPI backend + SQLite jobs | **Tested** | Migrated from Starlette to FastAPI (`OpenAPI 3.1`), multipart `/api/v1/documents/upload`, SQLite `jobs` table + SSE `/api/v1/jobs/{id}/events` |
| React + TypeScript + Tailwind + shadcn-style workspace | **Implemented / Tested (14/14 pages)** | All 14 pages (`Workspace`, `Upload`, `Benchmark`, `Timeline`, `RelationGraph`, `ReviewQueue`, `PdfViewer`, `Ablation`, `ErrorTaxonomy`, `CorpusExplorer`, `Reports`, `SchemaExplorer`, `SettingsDiagnostics`, `Login`), Recharts, `pdfjs-dist`, Vitest (7/7 passing) |
| EN/FR traceable reports | **Tested** | `medresearch_platform/reports/generator.py` (Markdown, HTML, PDF with negation badges, source `[doc, p., chars]` citations, EN/FR disclaimer), `/api/v1/documents/{id}/report`, `medextract report` |
| Final scientific evaluation & `RESEARCH_REPORT.md` | **Evaluated on `SYNTH-MED-V3` (1,240 docs; 480 locked test docs)** | Honest `docs/RESEARCH_REPORT.md` published from `docs/results/v3_*.json`; v2 report archived in `docs/results/archive/2026-11-14_synth_v2_buggy/` |

---

## 3. Questions for the Owner

1. **NumPy vs PyTorch (`DECISIONS/ADR-0009-numpy-vs-pytorch.md`):** Currently the neural models run in pure NumPy because Python 3.13 on macOS `x86_64` has no official `torch` wheel. Installing Python 3.12 (e.g. via `uv python install 3.12` or `pyenv`) enables `torch 2.2.2` CPU on Intel Mac (and newer `torch` on Windows/Linux). Do you want to stay on pure NumPy (zero C++ runtime issues, slower training, smaller models) or switch to Python 3.12 + PyTorch CPU? Keeping NumPy until you decide.
2. **Realistic Free-Text Evaluation Set (Phase 11 Step 2.6):** After Steps 1–2, will you author/annotate 40 EN + 40 FR free-text notes (using our Label Studio protocol) or provide a licensed public corpus?
