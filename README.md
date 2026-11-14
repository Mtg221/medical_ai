# Multilingual Medical Information Extraction AI (`medical_ai`)

**Version:** `0.1.0` (Phase 0 — Foundation & Honesty Reset)  
**Primary Languages:** English (`en`) & French (`fr`)  
**Execution Profile:** CPU-Only Native (`Intel Core i7-9750H`, 32 GB RAM, macOS Darwin `x86_64`)  
**Primary AI Track:** Models trained strictly from scratch (random initialization, zero pretrained weights)

> **Clinical & Scientific Safety Notice**  
> This platform is a **research tool — not for diagnosis or treatment; human review is required before any clinical use**. Do not use AI-extracted or inferred outputs as verified clinical truth.

---

## 1. Current Baseline Reality (Audited 2026-11-14)

The repository is undergoing a structured rebuild (see [docs/AUDIT.md](docs/AUDIT.md) and [docs/MASTER_PROMPT.md](docs/MASTER_PROMPT.md)).

Current six-approach extraction benchmark (`run_six_approach_benchmark(epochs=20)` on `SYNTH-MED-EN-FR`, **3 train documents / 1 English test document**):

| Approach | Precision | Recall | F1 | FR F1 | Notes |
|---|---:|---:|---:|---:|---|
| 1. Rule-based | 0.53 | 1.00 | 0.69 | 0.00 | Shared cue-lexicon assertions |
| 2. TF-IDF + LogReg | 0.75 | 0.33 | 0.46 | 0.00 | Token baseline |
| 3. BiLSTM (NumPy) | 0.00 | 0.00 | **0.00** | 0.00 | Underperforming on 3-doc train |
| 4. BiLSTM-CRF (NumPy) | 0.57 | 0.44 | 0.50 | 0.00 | |
| 5. Transformer (NumPy) | 0.00 | 0.00 | **0.00** | 0.00 | Underperforming on 3-doc train |
| 6. Hybrid | 0.43 | 1.00 | 0.60 | 0.00 | Currently weighted vote, not 5-stage pipeline |

These numbers reflect a **4-document synthetic prototype** and are **not** a validated scientific result. Expanding the corpus (Phase 1), building a rigorous evaluation harness (Phase 2), and rebuilding the 5 single approaches + 5-stage Hybrid pipeline in PyTorch CPU (Phases 3–4) are the next milestones.

---

## 2. Quick Start

```bash
# 1. Install package (editable) and dev tools
pip install -e ".[dev]"

# 2. Build frontend assets (required for /react workspace route)
npm --prefix frontend ci && npm --prefix frontend run build

# 3. Seed demo users (set an explicit password via environment variable)
export MEDRESEARCH_DEMO_PASSWORD="change-me-local-only"
python3 -m medresearch_platform.interfaces.cli init-db

# 4. Run tests
pytest -q
```

---

## 3. Documentation & ADRs

- [docs/MASTER_PROMPT.md](docs/MASTER_PROMPT.md) — Project roadmap, constraints, and phase definitions
- [docs/AUDIT.md](docs/AUDIT.md) — Verified baseline audit (2026-11-14)
- [docs/EXTERNAL_COMPONENTS.md](docs/EXTERNAL_COMPONENTS.md) — Allowed non-model utilities (Tesseract, PyMuPDF, pypdf, python-docx)
- [PROJECT_STATUS.md](PROJECT_STATUS.md) — Honest phase & capability status
- [ARCHITECTURE.md](ARCHITECTURE.md) — System architecture overview
- [DECISIONS/](DECISIONS/) — Architecture Decision Records (`ADR-0001` to `ADR-0008`)
- [extras/](extras/) — Frozen non-core modules (dermatology CV, multimodal vision fusion) isolated per `ADR-0008`
