# System Architecture — Multilingual Medical Information Extraction AI (`medical_ai`)

**Version:** `0.2.0`  
**Architecture Style:** Strict Modular Monolith (CPU-First, Zero-Trust, Strictly From-Scratch Primary AI Track)  
**Target Hardware:** Intel CPU (`x86_64`), 32 GB RAM, CPU-only execution ([`ADR-0001`](DECISIONS/ADR-0001-modular-monolith-and-cpu-first-architecture.md), [`ADR-0009`](DECISIONS/ADR-0009-numpy-vs-pytorch.md), [`ADR-0010`](DECISIONS/ADR-0010-pytorch-cpu-for-vision.md))

---

## 1. High-Level System Architecture

```mermaid
flowchart TD
    subgraph Interfaces["Interfaces & Dashboard (medresearch_platform.interfaces + frontend/)"]
        CLI["CLIs: medextract & medresearch-cli"]
        API["FastAPI App Factory (api.py)"]
        ROUTERS["Domain Routers (routers/{system,documents,reviews,experiments,vision}.py)"]
        UI["React 19 + TypeScript + Tailwind v4 Dashboard (frontend/, 16 pages)"]
    end

    subgraph Security["Zero-Trust Security & Privacy (medresearch_platform.security)"]
        AUTH["Argon2id Auth + HMAC Token RBAC/ABAC (auth.py)"]
        SAFE["Untrusted Upload & Magic-Byte/Zip-Bomb Guard (safety.py)"]
        PRIV["Bilingual PII Scrubber & Person/Age Pre-Filter (privacy.py, person_age_detector.py)"]
        AUD["Tamper-Evident SHA-256 Hash-Chained Audit Log (audit.py)"]
    end

    subgraph Documents["Document Processing & OCR (medresearch_platform.documents)"]
        PARSE["TXT / Digital & Scanned PDF / DOCX / Image Parsers (parsers.py, tables.py)"]
        OCR["Tesseract EN/FR OCR + Column-Split & Char-Level Span Alignment (ocr_engine.py, ocr_benchmark.py)"]
    end

    subgraph NLP["From-Scratch Multilingual NLP Track (pure NumPy + scikit-learn + CRFsuite)"]
        APP["Single Approaches (approaches/): Rule-Based, TF-IDF+SVM, CRF, BiLSTM, BiLSTM-CRF, BPE Transformer"]
        HYB["5-Stage Hybrid Pipeline (hybrid/): S1 Doc SVM -> S2 BiLSTM-CRF+CRF+Rules -> S3 CNN Rel + ConText/BiLSTM Assert -> S4 CRF TIMEX + EN/FR Normalizer -> S5 OOF Transformer Reranker + Calibration + Validation"]
        SERVE["Model Registry & Inference Facade (serving/extractor.py)"]
        CONTRACT["Pydantic v2 Output Contract & JSON Schema (extraction/schema_models.py)"]
    end

    subgraph Vision["Dermatology Skin-Image Vision Track (medresearch_platform.vision, optional .[vision])"]
        VROUTE["5-Class Picture Triage Router + Dense Text-Line Guard (router_torch.py, quality.py)"]
        VSKIN["4-Stage From-Scratch PyTorch ResNet + Calibration + Abstention (cnn_torch.py, serving.py)"]
        VLINK["Image-Text Same-Lesion Linking & Unified Patient Record (linking.py)"]
    end

    subgraph EvalReports["Evaluation, IAA & Traceable Reports"]
        EVAL["Evaluation Harness: 5-Seed Runner, Bootstrap CIs, Holm-Bonferroni, IAA & Profiler (eval_harness/)"]
        RPT["Bilingual EN/FR Traceable Markdown/HTML/PDF Reports (reports/generator.py)"]
    end

    subgraph Storage["Workflows & Storage (medresearch_platform.workflows + db)"]
        PIPE["MedicalResearchPlatformService, SQLite Jobs/SSE & Review/Gold Export (workflows/)"]
        DB["DatabaseEngine: SQLite WAL + PostgreSQL DDL Migrations (db/)"]
    end

    UI --> API
    CLI --> PIPE
    API --> ROUTERS
    ROUTERS --> AUTH
    ROUTERS --> PIPE
    PIPE --> SAFE
    PIPE --> PRIV
    PIPE --> PARSE
    PARSE --> OCR
    PIPE --> SERVE
    SERVE --> HYB
    SERVE --> APP
    HYB --> CONTRACT
    PIPE --> VROUTE
    VROUTE --> VSKIN
    VSKIN --> VLINK
    PIPE --> RPT
    PIPE --> AUD
    PIPE --> DB
    EVAL --> APP
    EVAL --> HYB
```

---

## 2. Package Boundaries & Responsibilities

1. **`medresearch_platform.config` & `diagnostics`:** Environment configuration, CPU/memory inspection, and readiness diagnostics with zero internal dependencies.
2. **`medresearch_platform.data`:** Canonical annotation schema (`EntitySpan`, `RelationSpan`, `TimexSpan`, `AnnotatedDocument`), `SYNTH-MED-V3.1` generator, and SHA-256 locked split builder (`builder.py`).
3. **`medresearch_platform.documents`:** Native parsers for TXT, digital/scanned PDF (`pypdf`, `pdfplumber`, `pypdfium2`), DOCX (`python-docx`), and raster images, plus Tesseract (`eng` + `fra`) OCR with two-column layout segmentation and character-level reference span mapping.
4. **`medresearch_platform.approaches`:** Independently implemented single-paradigm EN/FR models trained strictly from random initialization in pure vectorized NumPy / `scikit-learn` / `sklearn-crfsuite`: `rule_based`, `classical_ml`, `classical_crf`, `bilstm`, `bilstm_crf`, and stacked multi-head BPE `transformer`, plus bilingual `context_negex.py` assertion/scope rules.
5. **`medresearch_platform.hybrid`:** Five-stage pipeline (`stage1_doc_understanding.py` through `stage5_integration.py`) with SHA-256 manifest persistence (`persistence.py`), out-of-fold Stage 5 Transformer reranking and temperature calibration, general EN/FR ISO-8601 TIMEX normalization, clinical consistency checks, and full ablation toggles.
6. **`medresearch_platform.extraction` & `reports`:** Single-source-of-truth Pydantic v2 JSON contract (`schema/medical_extraction.schema.json`) with source offsets, assertion/uncertainty flags, and traceable bilingual Markdown/HTML/PDF report generation.
7. **`medresearch_platform.eval_harness`:** Strict/lenient entity, assertion, relation, TIMEX, doc-type, and calibration metrics, 10,000-resample document-level bootstrap CIs, Holm-Bonferroni significance testing, error taxonomy, resource profiler, and double-annotation Inter-Annotator Agreement (`iaa.py`).
8. **`medresearch_platform.vision`:** Optional `.[vision]` PyTorch CPU dermatology track (`ADR-0010`) with 5-class picture triage router, illumination-invariant `detect_dense_text_lines` document safety guard, lesion-grouped splits, calibrated risk bands (dermoscopic-only served risk bands; clinical phone photos and TBP tiles abstain/suppress), and image-text linking.
9. **`medresearch_platform.security`:** Argon2id password hashing, HMAC Bearer tokens, RBAC/ABAC + institution isolation, untrusted file validation, bilingual PII redaction (`privacy.py`, `person_age_detector.py`), and hash-chained audit logs (`audit.py`).
10. **`medresearch_platform.workflows`, `db`, `serving`, `interfaces`:** SQLite WAL / PostgreSQL persistence, chunked background jobs with SSE progress, human review & `jsonl`/`brat` gold export, modular FastAPI routers (`interfaces/routers/`), and the React 19 + TypeScript + Tailwind dashboard (`frontend/`).
11. **`extras/legacy/`:** Frozen legacy prototypes isolated from the active package per [`ADR-0008`](DECISIONS/ADR-0008-scope-reduction-and-extras-isolation.md).
