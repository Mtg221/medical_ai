# Project Status

**Last Updated:** 2026-11-15  
**Maturity:** **Research prototype, evaluated on synthetic data only; free-text evaluation pending owner notes** (see [`docs/REALISTIC_SET_PROTOCOL.md`](docs/REALISTIC_SET_PROTOCOL.md) and [`docs/REALISTIC_SET_OWNER_CHECKLIST.md`](docs/REALISTIC_SET_OWNER_CHECKLIST.md)).

Status vocabulary: `Planned` | `In progress` | `Implemented` | `Tested` | `Evaluated on <dataset, n docs>`

---

## 1. Machine & Environment Baseline

- **OS:** macOS Darwin `x86_64` (CI matrix: `ubuntu-latest`, `windows-latest`, `macos-13`)
- **CPU:** Intel(R) Core(TM) i7-9750H @ 2.60GHz (6 cores / 12 threads)
- **RAM:** 32 GB (CPU-only execution)
- **Python:** 3.12 & 3.13 | **Node.js:** v24.18.0

---

## 2. Component Status Matrix

| Component | Status | Notes |
|---|---|---|
| Packaging, CI, ADRs 0005–0009, `extras/` isolation | **Implemented / Tested** | Fresh-clone pytest fixture (`tests/conftest.py`), cross-platform CI matrix (`ubuntu-latest`, `windows-latest`, `macos-13` × Python `3.12`/`3.13`), `file:///` doc link guard, `extras/legacy/` isolated (`ai/nlp_model.py`, `ai/ocr_model.py`) |
| Document parsers & OCR (TXT, digital/scanned PDF, DOCX, PNG/JPEG/TIFF/WEBP) | **Evaluated on `SYNTH-MED-V3.1` `test_iid` (120 pages = 60 EN + 60 FR)** | Character-level `map_ocr_spans_to_reference` + `detect_column_split`: overall CER `0.0862` (`en` `0.0725`, `fr` `0.0999`), WER `0.2112`; `classical_crf` Strict F1 `0.9978 -> 0.9128` (`-0.0850`), `hybrid` `0.9937 -> 0.8799` (`-0.1139`) (`docs/results/v3_1_ocr_full_benchmark.json`) |
| Bilingual terminology registry (EN/FR) | **Tested** | Static registry; corpus-scale normalization evaluation pending |
| Security (Argon2id auth, HMAC tokens, PII scrubber, audit chain) | **Tested** | Uses `MEDRESEARCH_DEMO_PASSWORD` env var |
| Output contract (`extraction/contract.py` & `extraction/schema_models.py`) | **Tested** | Pydantic v2 single-source-of-truth + exported `schema/medical_extraction.schema.json` + `/api/v1/schema/extraction` |
| Annotated EN/FR synthetic corpus (`>= 300 EN + >= 300 FR`) | **Implemented / Audited (`SYNTH-MED-V3.1`, 1,240 docs)** | `SYNTH-MED-V3.1` (`data/processed/v3_1/`): 620 EN + 620 FR, 6 structural doc types without title-header leakage, `train` (600), `val` (160), `test_iid` (160), `test_unseen_terms` (160, 0% term overlap), `test_unseen_templates` (160, 0.6% masked overlap) |
| Realistic free-text evaluation tooling | **Implemented / Waiting on owner notes** | Free-text evaluation: not done yet — waiting for owner-written notes (`configs/label_studio_medical_en_fr.xml`, `scripts/import_label_studio_realistic.py`, `scripts/eval_realistic.py`, `docs/REALISTIC_SET_OWNER_CHECKLIST.md`). Invalid template-generated `realistic_80` archived (`docs/results/archive/2026-11-15_realistic_v1_invalid/`). |
| Evaluation harness (CI, 10k bootstrap, Holm-Bonferroni, error taxonomy, profiler) | **Tested (`SYNTH-MED-V3.1` 5-seed suite)** | 10,000-resample 160-document bootstrap 95% CIs (`bootstrap_mean_seed_f1_ci` matching point estimates) + separate seed-std column + 21-pair Holm-Bonferroni tables |
| 5 Single Approaches (Rule, TF-IDF+ML/CRF, BiLSTM, BiLSTM-CRF, Transformer) | **Evaluated on `SYNTH-MED-V3.1` (all 4 splits, seeds 1–5)** | Per-task NER leaders: `classical_ml` on `test_iid` (`0.9983`) & `test_unseen_templates` (`0.9670`); `classical_crf` on `test_unseen_terms` (`0.9169`) + `save`/`load` (`models/classical_crf_seed1/`) |
| 5-Stage Hybrid (`v1` & `val`-selected `v2`) | **Evaluated & Served (`models/hybrid_v2_seed1/`)** | `hybrid_v2` (`val`-selected `crf_on_ctx_nofusion`) served via `ExtractionService` (`138.1s` CPU train via `medextract train`): `0.9972` `test_iid`, `0.9204` `test_unseen_terms`, `0.9310` `test_unseen_templates`; API-vs-pipeline consistency verified on 30 TXT + 10 PDF docs |
| FastAPI backend + SQLite jobs & reviews | **Tested (Phase 1 P1–P9 hardened)** | FastAPI OpenAPI 3.1, `401`/`403` Bearer middleware on all `/api/v1/*` routes, `415` magic-byte allow-list, `413` size limits, duplicate `content_sha256` detection (`duplicate_of`), chunked worker jobs with stage progress (`PARSING`..`VALIDATE`), document-date text extraction (`P6`), `/api/v1/models` provenance, `/api/v1/reviews` persistence + `jsonl`/`brat` export |
| React + TypeScript + Tailwind + shadcn-style workspace | **Implemented / Tested (14/14 pages + E2E + Lighthouse 100)** | Shared `entityStyles.ts` (type colours + assertion styles + legend), fixed Relation Graph + node-click jump, Settings model provenance, EN/FR Playwright E2E, 28 screenshots (`14` pages × light/dark), Lighthouse `100/100` on 6 routes (`docs/results/lighthouse_accessibility_report.json`) |
| EN/FR traceable reports | **Tested** | `medresearch_platform/reports/generator.py` (Markdown, HTML, PDF with negation badges, source citations, EN/FR disclaimer), `/api/v1/documents/{id}/report`, `medextract report` |
| Final scientific evaluation & `RESEARCH_REPORT.md` | **Evaluated on `SYNTH-MED-V3.1` (synthetic only)** | Regenerated from verified JSON results in `docs/RESEARCH_REPORT.md` with H1–H9 remediation summary |

---

## 3. Questions / Pending Input from the Owner

1. **NumPy vs PyTorch (`DECISIONS/ADR-0009-numpy-vs-pytorch.md`):** Currently the neural models run in pure NumPy because Python 3.13 on macOS `x86_64` has no official `torch` wheel. Keeping NumPy until you decide otherwise.
2. **Owner-Written Realistic Free-Text Notes (`docs/REALISTIC_SET_OWNER_CHECKLIST.md`):** Ready for 40 EN + 40 FR de-identified/owner-written notes annotated in Label Studio (`configs/label_studio_medical_en_fr.xml`) and imported via `scripts/import_label_studio_realistic.py`.
