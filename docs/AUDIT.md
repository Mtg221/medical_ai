# Repository Audit (`medical_ai` / `medresearch_platform`)

**Audit Date:** 2026-11-14  
**Scope:** Verification of Part A.3 of `MASTER_PROMPT.md` against the actual codebase, test suite, and benchmark execution.

---

## 1. Executive Summary

The repository (~12,060 lines across `medresearch_platform/`, `frontend/src/`, and `tests/`, plus documentation) provides a working modular-monolith skeleton with a common extraction contract, six extractor interfaces, Starlette API, React prototype, and security/audit primitives. However, **the claims in `README.md` and `PROJECT_STATUS.md` are substantially ahead of the measured experimental reality**:

- The extraction benchmark trains on **3 documents** and evaluates on **1 English document** (`SYNTHETIC_CLINICAL_CORPUS`: 4 documents total).
- French (`fr`) extraction is **not evaluated** (`fr_f1 = 0.0` for all 6 approaches).
- Two neural approaches (**BiLSTM** and **Transformer**) score **0.0000 F1** on the benchmark.
- The **Hybrid** approach scores **0.6000 F1**, below the **Rule-based** baseline (**0.6923 F1**).
- Several engineering and scope issues block clean installation, security hygiene, and maintainability.

---

## 2. Verified Test Suite Status

Command executed: `pytest -q`

```text
26 passed, 1 warning in 11.48s
```

**Important nuance:**
- In this local workspace, `26/26` tests pass **only because `frontend/dist/` is already built on disk**.
- `frontend/dist/` is excluded by `.gitignore` (not committed to git). On a fresh clone without running `npm run build` inside `frontend/`, `test_frontend_assets_and_security_headers` will fail with HTTP 404 on `/react`.

---

## 3. Verified Six-Approach Benchmark (`run_six_approach_benchmark(epochs=20)`)

Command executed:
```bash
python3 -c "from medresearch_platform.extraction.benchmark import run_six_approach_benchmark; import json; res = run_six_approach_benchmark(epochs=20); print(json.dumps(res, indent=2, default=str))"
```

Measured output (`corpus="SYNTH-MED-EN-FR"`, `train_documents=3`, `test_documents=1`, `leakage_verified=True`, `pretrained_weights_used=False`):

| Approach | Model Class | Precision | Recall | F1 | Partial F1 | EN F1 | FR F1 | Train (s) | Inf (ms/doc) | Params |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1. Rule-based | `RuleBasedMedicalExtractor` | 0.5294 | 1.0000 | **0.6923** | 0.6923 | 0.0000 | 0.0000 | 0.000 | 0.999 | 9 |
| 2. TF-IDF + ML | `ClassicalTFIDFExtractor` | 0.7500 | 0.3333 | **0.4615** | 0.6154 | 0.0000 | 0.0000 | 0.059 | 3.106 | 19,212 |
| 3. BiLSTM | `RandomInitBiLSTMExtractor` | 0.0000 | 0.0000 | **0.0000** | 0.0000 | 0.0000 | 0.0000 | 0.484 | 6.423 | 22,931 |
| 4. BiLSTM-CRF | `RandomInitBiLSTMCRFExtractor` | 0.5714 | 0.4444 | **0.5000** | 0.5000 | 0.0000 | 0.0000 | 0.753 | 7.163 | 23,330 |
| 5. Transformer | `RandomInitTransformerExtractor` | 0.0000 | 0.0000 | **0.0000** | 0.3333 | 0.0000 | 0.0000 | 0.117 | 1.936 | 18,323 |
| 6. Hybrid | `HybridMultimodalPipeline` | 0.4286 | 1.0000 | **0.6000** | 0.6667 | 0.0000 | 0.0000 | 0.812 | 11.613 | 42,551 |

**Key observations:**
- All numbers match Part A.3 of the Master Prompt.
- `en_f1` and `fr_f1` both report `0.0` (the single test document is not counted under `en`/`fr` properly or only one language is present in test).
- `Exact F1 = 0.9610` cited in `PROJECT_STATUS.md` comes from `CustomMedicalNLPNet-v1` on the same 4-document synthetic corpus and is not representative of real generalization.

---

## 4. Verification of Architecture & Engineering Findings

| Finding | Verification Result | Evidence in Codebase |
|---|---|---|
| **4-document corpus** | Confirmed | `medresearch_platform/documents/synthetic_generator.py`: `SYNTHETIC_CLINICAL_CORPUS` contains 4 `SyntheticDocumentSpec` entries (2 train, 1 val, 1 test). |
| **Neural NLP model is regex + small MLP; shared assertion cue-lexicon** | Confirmed | `medresearch_platform/ai/nlp_model.py` uses regex/lexicon candidate matching + MLP; `medresearch_platform/extraction/text.py::assess_assertion` is shared across all 6 extraction approaches. |
| **Hybrid is not the 5-stage/6-technique target architecture** | Confirmed | `medresearch_platform/extraction/hybrid_multimodal.py` runs weighted voting over `RuleBasedMedicalExtractor` (0.45), `ClassicalMLExtractor`, and `BiLSTMCRFExtractor`. Missing: TF-IDF+SVM doc classifier, BiLSTM assertion classifier, CNN relation extractor, CRF temporal tagger, Transformer fusion/reranker, consistency checker. |
| **OCR is synthetic glyph patch classification** | Confirmed | `medresearch_platform/ai/ocr_model.py` (`CustomOCRConvNet`) classifies rendered glyph patches; cannot OCR real scanned PDFs/images. |
| **Scope creep** | Confirmed | `medresearch_platform/dermatology/`, federated learning & DP accounting (`ai/evaluation.py`), `CustomMedicalVisionNet` / `CustomMultimodalFusionNet` (`ai/vision_multimodal.py`), and `v23.2–v40.0` references dilute the core NLP extraction mission. |
| **Packaging fails (`pip install -e .`)** | Confirmed | `pyproject.toml` lacks explicit `[tool.setuptools.packages.find]` include rules, and `pip install -e . --no-deps --no-build-isolation --dry-run` fails in the current environment (`BackendUnavailable: Cannot import 'setuptools.build_meta'`). |
| **Hard-coded demo password in UI & backend** | Confirmed | `frontend/src/App.jsx` lines 196 & 228 pre-fill and display `"MedResearch!2026"`; `medresearch_platform/workflows/pipeline.py` line 115 hardcodes `hash_password("MedResearch!2026")`. |
| **Absolute local file links in `README.md`** | Confirmed | `README.md` lines 73–83 use `file:///Users/guest777/Desktop/medical/...` links. |
| **Duplicate frontends & monolithic `App.jsx`** | Confirmed | Both `medresearch_platform/interfaces/static/` (vanilla JS) and `frontend/src/App.jsx` (894 lines, plain CSS) exist. |
