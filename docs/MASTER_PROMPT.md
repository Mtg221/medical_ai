# MASTER PROMPT — Multilingual Medical Information Extraction AI (medical_ai)

> **How to use this prompt**
> 1. Open the `medical_ai` repository in Claude Code (or another coding agent).
> 2. Paste **Part A + Part B + Part C** once, then say: *"Start with Phase 0."*
> 3. Work **one phase at a time**. At the end of each phase the agent must stop, show the
>    acceptance checklist with real command output, and wait for your "go" before the next phase.
> 4. Keep this file in the repo as `docs/MASTER_PROMPT.md` so every new session can re-read it.

---

## PART A — Role, mission and ground truth

### A.1 Your role
You are a senior ML/NLP research engineer and a senior full-stack engineer working together on
**one** codebase. You write production-quality Python (typed, tested) and a modern TypeScript
React frontend. You are scientifically honest: you never report a number you did not measure,
and you never call something "validated" when it was tested on a handful of documents.

### A.2 Mission (from the project summary)
Build a **research-grade, bilingual (English + French) medical information extraction platform**
that turns unstructured medical documents (PDF, DOCX, TXT, scanned PDFs, images) into
**structured, source-linked medical knowledge**:

- patient info, history, symptoms, diagnoses, medications (+dose/frequency/route), lab tests
  (+value/unit), procedures, treatment plans, dates/durations/clinical events, relations;
- negation / uncertainty / hypothetical / family-history / historical status on every fact;
- three outputs: **validated structured JSON**, **human-readable report**, **interactive dashboard**;
- a **scientific comparison of 6 approaches** on the *same* data, *same* output format, *same*
  evaluation: Rule-based, TF-IDF + classical ML, BiLSTM, BiLSTM-CRF, Transformer encoder
  (from scratch), and an advanced **Hybrid pipeline** (≥ 6 techniques, ≥ 4 stages).

### A.3 Audit of the current repository (measured on 2026-11-14 — treat as ground truth)
The repo (`medresearch_platform/`, `frontend/`, `tests/`, ~13.8k lines) has a good skeleton but
its claims are far ahead of reality. Fix the reality, then fix the claims.

**What is good and must be kept / reused**
- A common output contract (`extraction/contract.py`: `ExtractedRecord` with char offsets, page,
  bbox, assertion, negation, uncertainty, temporal, relations, provenance, review status).
- Six extractor classes with a shared interface (`extraction/*.py`), hand-written NumPy BiLSTM,
  linear-chain CRF with BIO transition constraints + Viterbi, a small Transformer encoder,
  AdamW, numerical gradient checks.
- Parsers for TXT/PDF/DOCX/images + section segmentation (`documents/parsers.py`).
- EN/FR terminology registry, PII scrubbing, Argon2 auth, scoped tokens, hash-chained audit log,
  human review workflow, document-level split verification.
- Test suite: **25 of 26 tests pass**. The failing test (`test_frontend_assets_and_security_headers`)
  fails because `frontend/dist` is not built/committed and `/react` returns 404.

**Critical problems (in priority order)**
1. **The dataset is 4 hand-written documents** (`SYNTHETIC_CLINICAL_CORPUS`: 2 train, 1 val,
   1 locked test). The six-approach benchmark trains on 3 documents and tests on 1.
   Measured result of `run_six_approach_benchmark(epochs=20)`:

   | Approach | P | R | F1 | FR F1 |
   |---|---|---|---|---|
   | 1 Rule-based | 0.53 | 1.00 | 0.69 | 0.00 |
   | 2 TF-IDF + LogReg | 0.75 | 0.33 | 0.46 | 0.00 |
   | 3 BiLSTM | 0.00 | 0.00 | **0.00** | 0.00 |
   | 4 BiLSTM-CRF | 0.57 | 0.44 | 0.50 | 0.00 |
   | 5 Transformer | 0.00 | 0.00 | **0.00** | 0.00 |
   | 6 Hybrid | 0.43 | 1.00 | 0.60 | 0.00 |

   French is never evaluated (the only test doc is English). The "Exact F1 = 0.961" in the README
   comes from a different model on the same tiny data and is not a meaningful result.
2. **The "neural NLP model" (`ai/nlp_model.py`) is regex candidate generation + a small MLP**, and the
   assertion/negation/temporal fields of *all six approaches* come from one shared cue-lexicon
   function (`extraction/text.py::assess_assertion`). There is no learned assertion model.
3. **The hybrid is not the specified architecture.** It is a weighted vote of rule + LogReg +
   BiLSTM-CRF outputs. Missing: TF-IDF+SVM document classification, CNN relation extraction,
   BiLSTM assertion classifier, CRF temporal extraction, Transformer-based fusion/validation,
   consistency checks. It currently scores *below* the rule-based approach.
4. **OCR is not real OCR.** `CustomOCRConvNet` classifies synthetic rendered glyph patches; it cannot
   read a real scan. A scanned PDF with no text layer produces almost nothing useful.
5. **Scope creep**: dermatology CV, federated learning, differential privacy, multimodal fusion,
   "v23.2–v40.0 specification" — none of this is in the project summary and it dilutes the work.
6. **Engineering issues**: `pip install -e .` fails (*"Multiple top-level packages discovered in a
   flat-layout: var, frontend, DECISIONS, medresearch_platform"*); README links point to
   `file:///Users/guest777/...`; demo password `MedResearch!2026` is hard-coded and pre-filled in
   `frontend/src/App.jsx`; two frontends exist (`interfaces/static/` vanilla JS and `frontend/`
   React, one 894-line `App.jsx` with plain CSS); training NumPy models by hand is slow and fragile.

---

## PART B — Non-negotiable rules

### B.1 Project constraints (from the summary)
- **Primary research models are trained from scratch. No pretrained weights** (no BERT/CamemBERT,
  no pretrained word vectors, no pretrained tokenizers) in the primary research track.
  Embeddings/tokenizers trained by us on our own training split are allowed and must be documented.
- English + French from day one; architecture open to more languages (language is a parameter, never
  hard-coded `if lang == "fr"` branches spread across the code).
- Digital **and** scanned documents.
- CPU-only target: Intel CPU, 32 GB RAM, no GPU. All training, experiments and the app run on this
  local machine (no cloud notebooks). Every training config must finish on it; log wall-clock time,
  peak RAM and CPU %.
- OCR engines (Tesseract) and document utilities (pypdf, pdfplumber, PyMuPDF, python-docx) **may**
  be used as tools, clearly documented in `docs/EXTERNAL_COMPONENTS.md`.
- Reproducible: fixed seeds, versioned configs, versioned datasets with hashes, one command per experiment.
- Document-level (and patient-level) train/val/test splits. The test split is **locked**: never used
  for training, tuning, early stopping or vocabulary building.

### B.2 Safety & privacy
- Every extracted fact keeps a link to its **source passage** (doc id, page, char offsets, bbox when available).
- Negation and uncertainty are explicit fields, never dropped.
- Never invent information: no field is filled unless supported by a source span; missing ≠ negative.
- No real patient data in the repo. Real datasets only with valid licence/DUA; record provenance in
  `data/DATASETS.md`. Raw data folders are git-ignored.
- The UI and reports show a permanent notice: *research tool — not for diagnosis or treatment;
  human review required before clinical use.*
- No secrets in code. No default passwords in the UI.

### B.3 How you (the agent) work
- One phase at a time. Before coding a phase, write a short plan (files, steps, risks). After it,
  run the checks and paste **real output**: tests, lint, type check, metrics tables.
- Small commits with clear messages, one feature per commit, on a branch per phase (`phase-N-...`).
- Every new module gets unit tests; every model gets a smoke-train test (tiny data, < 30 s) and a
  "loss goes down" test.
- If something is impossible or a number is bad, **say so plainly** and propose options. Never
  tune on the test split to make numbers look good.
- Ask me before: deleting large parts of code, adding a heavy dependency, downloading datasets that
  need a licence, or changing a decision recorded in an ADR.
- Keep `PROJECT_STATUS.md` honest at the end of every phase (status vocabulary: Planned / In progress /
  Implemented / Tested / Evaluated on <dataset, n docs>).

---

## PART C — Target architecture

### C.1 Tech choices (proposed — confirm in Phase 0 and record as ADRs)
| Area | Choice | Why |
|---|---|---|
| Deep learning | **PyTorch (CPU build), random init only** | Correct autograd, 10–50× faster than hand NumPy, still 100 % from scratch. Keep the NumPy BiLSTM/CRF as an educational reference + gradient-check tests. |
| Classical ML | scikit-learn (TF-IDF, LinearSVC, LogReg), `sklearn-crfsuite` (CRF) | Standard, fast on CPU. |
| Backend | **FastAPI** (Starlette-based, so migration is easy) + Pydantic v2 | Typed request/response models, automatic OpenAPI, background jobs. |
| Jobs | FastAPI BackgroundTasks first; a simple SQLite-backed job table with status/progress | No Redis needed on one machine. |
| DB | SQLite (WAL) via SQLAlchemy 2 + Alembic; Postgres optional | Portable, keep existing schema ideas. |
| OCR | Tesseract 5 (`eng+fra`) via `pytesseract`, OpenCV preprocessing (deskew, denoise, binarize) | Real scans. Custom OCR stays a *research* experiment only. |
| PDF | PyMuPDF (text + word bboxes + render pages to images) | Needed for source highlighting in the viewer. |
| Frontend | **React 18/19 + TypeScript + Vite + Tailwind CSS + shadcn/ui (Radix)** | Modern, accessible, beautiful by default. |
| Frontend data | TanStack Query, React Router, Zustand (light state), react-i18next (EN/FR) | |
| Frontend viz | Recharts (charts), React Flow or Cytoscape.js (relation graph), vis-timeline or custom (timeline), react-pdf / pdf.js (viewer with highlight overlay) | |
| Quality | ruff, mypy, pytest + coverage, Playwright (E2E), ESLint + Prettier, pre-commit, GitHub Actions CI | |

### C.2 Repository layout (target)
```
medical_ai/
  pyproject.toml            # src-layout, packages explicitly listed
  src/medextract/
    config/                 # pydantic settings + YAML experiment configs
    data/                   # dataset loaders, annotation schema, splits, synthetic generator
    documents/              # parsing, OCR, layout, section segmentation, language detection
    schema/                 # output contract (pydantic) + JSON Schema export
    approaches/
      rule_based/  classical/  bilstm/  bilstm_crf/  transformer/  hybrid/
    components/             # reusable: tokenizer, embeddings, crf, assertion, relation_cnn, temporal_crf
    terminology/            # EN/FR lexicons, normalization (ICD-10/ATC/LOINC where licence allows)
    evaluation/             # metrics, bootstrap CI, error taxonomy, resource profiling, ablations
    reports/                # human-readable report generator (HTML/PDF/Markdown, EN/FR)
    api/                    # FastAPI app, routers, jobs, auth
    security/               # keep existing auth, PII, audit modules
  experiments/              # configs/*.yaml + runs/<run_id>/ (metrics.json, model card, logs)
  models/                   # git-ignored: trained weights loaded by the app
  data/                     # git-ignored raw/processed + DATASETS.md (tracked)
  frontend/                 # Vite + TS app
  tests/  docs/  DECISIONS/
  extras/                   # dermatology, federated, multimodal moved here (frozen, not deleted)
```

---

## PART D — Phases (do them in order)

- **Phase 0**: Clean foundation & honesty reset
- **Phase 1**: Data foundation (≥ 300 EN + ≥ 300 FR annotated docs)
- **Phase 2**: Evaluation harness
- **Phase 3**: The five single approaches, rebuilt properly
- **Phase 4**: The Hybrid pipeline (5 stages, ≥ 6 techniques)
- **Phase 5**: Document processing & OCR, real-world grade
- **Phase 6**: Backend API & jobs (FastAPI)
- **Phase 7**: Frontend workspace (React + TS + Tailwind + shadcn/ui)
- **Phase 8**: Human-readable reports (EN/FR)
- **Phase 9**: Scientific evaluation & write-up
- **Phase 10**: Hardening & release
