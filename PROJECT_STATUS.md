# Project Status

**Last Updated:** 2026-11-14  
**Active Phase:** Phase 2 Complete — Evaluation Harness (`Evaluator`, `ResourceProfiler`, `error_taxonomy`, multi-seed runner)

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
| Document parsers (TXT, digital PDF, DOCX, image) | **Tested** | Digital extraction works; real scan OCR (Tesseract + PyMuPDF) planned in Phase 5 |
| Bilingual terminology registry (EN/FR) | **Tested** | Static registry; corpus-scale normalization evaluation pending |
| Security (Argon2id auth, HMAC tokens, PII scrubber, audit chain) | **Tested** | Hard-coded demo password removed; uses `MEDRESEARCH_DEMO_PASSWORD` env var |
| Output contract (`extraction/contract.py`) | **Implemented** | Pydantic v2 single-source-of-truth + JSON Schema export planned in Phase 6 |
| Annotated EN/FR corpus (`>= 300 EN + >= 300 FR`) | **Tested (`SYNTH-MED-V2`, 800 synthetic docs: 400 EN + 400 FR)** | `medextract data build --config configs/data/v1.yaml`; 70/15/15 patient-stratified splits (`555/124/121`), MinHash leakage check passed; BRAT & Label Studio converters ready; real clinical corpora (`n2c2`, `DEFT/CAS`) require DUA/agreement |
| Evaluation harness (CI, McNemar, error taxonomy, profiler) | **Tested** | `medresearch_platform/eval_harness/`; hand-made unit tests + `medextract experiment run configs/exp/dummy.yaml --seeds 1,2,3` (`Strict F1: 0.5249 ± 0.0000` on `SYNTH-MED-V2` val) |
| 5 Single Approaches (Rule, TF-IDF+ML/CRF, BiLSTM, BiLSTM-CRF, Transformer) | **Evaluated on `SYNTH-MED-EN-FR`, 4 docs** | BiLSTM & Transformer currently F1 = 0.00; FR F1 = 0.00; PyTorch rebuild in Phase 3 |
| 5-Stage Hybrid Pipeline | **Planned** | Current `HybridMultimodalExtractor` is a weighted vote (F1 = 0.60 on 1 test doc); Phase 4 |
| FastAPI backend + SQLite jobs | **Planned** | Currently Starlette prototype; Phase 6 |
| React + TypeScript + Tailwind + shadcn/ui workspace | **Planned** | Currently single-file `App.jsx` prototype; Phase 7 |
| EN/FR traceable reports | **Planned** | Phase 8 |
| Final scientific evaluation & `RESEARCH_REPORT.md` | **Planned** | Phase 9 |
