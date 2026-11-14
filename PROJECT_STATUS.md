# Project Status

**Last Updated:** 2026-11-14  
**Active Phase:** Phase 9 Complete — Locked Test-Split Evaluation (121 Docs, Seeds 1–3) & `docs/RESEARCH_REPORT.md`

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
| Annotated EN/FR corpus (`>= 300 EN + >= 300 FR`) | **Tested (`SYNTH-MED-V2`, 800 synthetic docs: 400 EN + 400 FR)** | `medextract data build --config configs/data/v1.yaml`; 70/15/15 patient-stratified splits (`555/124/121`), MinHash leakage check passed; BRAT & Label Studio converters ready; real clinical corpora (`n2c2`, `DEFT/CAS`) require DUA/agreement |
| Evaluation harness (CI, McNemar, error taxonomy, profiler) | **Tested** | `medresearch_platform/eval_harness/`; hand-made unit tests + `medextract experiment run configs/exp/dummy.yaml --seeds 1,2,3` (`Strict F1: 0.5249 ± 0.0000` on `SYNTH-MED-V2` val) |
| 5 Single Approaches (Rule, TF-IDF+ML/CRF, BiLSTM, BiLSTM-CRF, Transformer) | **Evaluated on `SYNTH-MED-V2` `val`, 124 docs (555 train)** | `rule_based` F1 `0.8708`, `classical_ml` F1 `0.8800`, `classical_crf` F1 `0.8800`, `bilstm` F1 `0.8800`, `bilstm_crf` F1 `0.8800`, `transformer` F1 `0.7598 ± 0.0206` (EN `0.8048` / FR `0.7605`); pure NumPy + `sklearn-crfsuite` per amended ADR-0005 |
| 5-Stage Hybrid Pipeline | **Evaluated on `SYNTH-MED-V2` `val`, 124 docs** | `NER F1 = 0.9362` (EN `0.9362` / FR `0.9362`), `+0.0562` vs `bilstm_crf` (`p = 0.00995`), `Assert = 0.8699`, `Rel F1 = 0.8477`, `Timex F1 = 0.3123`, `DocCls = 1.0000`, `ECE = 0.0258`; 6-variant ablation completed |
| FastAPI backend + SQLite jobs | **Tested** | Migrated from Starlette to FastAPI (`OpenAPI 3.1`), multipart `/api/v1/documents/upload`, SQLite `jobs` table + SSE `/api/v1/jobs/{id}/events` |
| React + TypeScript + Tailwind + shadcn-style workspace | **Tested** | Migrated from single `App.jsx` to modular TypeScript (`tsc --noEmit` + `vite build`), Tailwind CSS v4, TanStack Query, React Router, Zustand, i18next (EN/FR), source-linked viewer, multipart upload + job tracker |
| EN/FR traceable reports | **Tested** | `medresearch_platform/reports/generator.py` (Markdown, HTML, PDF with negation badges, source `[doc, p., chars]` citations, EN/FR disclaimer), `/api/v1/documents/{id}/report`, `medextract report` |
| Final scientific evaluation & `RESEARCH_REPORT.md` | **Evaluated on `SYNTH-MED-V2` locked `test`, 121 docs (seeds 1, 2, 3)** | `hybrid` Strict F1 `0.9362 ± 0.0000` (`+0.0562` vs `bilstm_crf`, `p = 0.003992`), `ECE = 0.0139`; full tables & error analysis in `docs/RESEARCH_REPORT.md` |
