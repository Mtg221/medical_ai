# Multilingual Medical Information Extraction AI (`medical_ai`)

**Version:** `0.1.0` (Phases 0–10 Complete)  
**Primary Languages:** English (`en`) & French (`fr`)  
**Execution Profile:** CPU-Only Native (`Intel Core i7-9750H`, 32 GB RAM, macOS Darwin `x86_64`)  
**Primary AI Track:** Models trained strictly from scratch (random initialization, zero pretrained weights)

> **Clinical & Scientific Safety Notice**  
> This platform is a **research tool — not for diagnosis or treatment; human review is required before any clinical use**. Do not use AI-extracted or inferred outputs as verified clinical truth.

---

## 1. Architecture & Verified Results (`SYNTH-MED-V2` Locked Test Split, 121 Docs, Seeds 1–3)

All primary models are trained from scratch on `train` (555 docs: EN + FR) and evaluated once on the locked `test` split (121 docs). Full methodology, ablations, OCR benchmarks, and limitations are documented in [docs/RESEARCH_REPORT.md](docs/RESEARCH_REPORT.md).

| Approach | Strict F1 (mean ± std) | EN F1 | FR F1 | Assert Acc | Rel F1 | TIMEX F1 | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1. `rule_based` | `0.8583 ± 0.0000` | `0.8673` | `0.8491` | `0.9936` | `0.8233` | `0.0000` | `0.0300` |
| 2. `classical_ml` (TF-IDF + LinearSVC) | `0.8800 ± 0.0000` | `0.8800` | `0.8800` | `0.9944` | `0.8696` | `0.0000` | `0.1441` |
| 3. `classical_crf` (`sklearn-crfsuite`) | `0.8800 ± 0.0000` | `0.8800` | `0.8800` | `0.9944` | `0.8696` | `0.0000` | `0.1188` |
| 4. `bilstm` (from-scratch NumPy) | `0.8798 ± 0.0003` | `0.8800` | `0.8800` | `0.9944` | `0.8696` | `0.0000` | `0.1198` |
| 5. `bilstm_crf` (from-scratch NumPy) | `0.8799 ± 0.0002` | `0.8800` | `0.8800` | `0.9944` | `0.8696` | `0.0000` | `0.0656` |
| 6. `transformer` (from-scratch NumPy) | `0.7634 ± 0.0180` | `0.8131` | `0.7473` | `0.9945` | `0.7537` | `0.1823` | `0.2058` |
| **7. `hybrid` (5-stage, 6-technique)** | **`0.9362 ± 0.0000`** | **`0.9362`** | **`0.9362`** | `0.8103` | `0.8329` | **`0.3071`** | **`0.0139`** |

Paired significance (`hybrid` vs `bilstm_crf`, seed 1): `ΔF1 = +0.05617`, `bootstrap_p_value = 0.003992`.

---

## 2. Quick Start

```bash
# 1. Install package (editable) and dev tools
pip install -e ".[dev]"

# 2. Build React + TypeScript + Tailwind frontend (served at /react)
npm --prefix frontend ci && npm --prefix frontend run build

# 3. Build dataset splits & run an experiment
medextract data build --config configs/data/v1.yaml
medextract experiment run configs/exp/hybrid.yaml --seeds 1,2,3

# 4. Extract structured JSON or EN/FR report from a document (TXT, PDF, DOCX, PNG)
medextract extract path/to/note.pdf --pipeline hybrid
medextract report path/to/note.pdf --lang fr --format html -o report_fr.html

# 5. Start the FastAPI server (set an explicit demo password for seeded local users)
export MEDRESEARCH_DEMO_PASSWORD="change-me-local-only"
uvicorn medresearch_platform.interfaces.api:create_app --factory --host 127.0.0.1 --port 8000

# 6. Run lint, type check, and tests
ruff check medresearch_platform tests && mypy medresearch_platform && pytest -q
```

### Tesseract OCR Setup (`eng` + `fra` Language Packs)

OCR tests automatically skip (`pytest.skip`) when `tesseract` or the `fra` language pack is missing.

- **macOS (Homebrew):**
  ```bash
  brew install tesseract tesseract-lang
  tesseract --list-langs   # verify eng and fra appear
  ```
- **Windows (winget / UB Mannheim installer):**
  ```powershell
  winget install UB-Mannheim.TesseractOCR
  # During setup (or via tessdata), select/download French (fra.traineddata) into C:\Program Files\Tesseract-OCR\tessdata\
  # Add C:\Program Files\Tesseract-OCR to PATH, then verify:
  tesseract --list-langs
  ```

---

## 3. Documentation & ADRs

- [docs/RESEARCH_REPORT.md](docs/RESEARCH_REPORT.md) — Locked test-split evaluation, ablations, OCR benchmark, and error analysis
- [docs/MASTER_PROMPT.md](docs/MASTER_PROMPT.md) — Project roadmap, constraints, and phase definitions
- [docs/EXTERNAL_COMPONENTS.md](docs/EXTERNAL_COMPONENTS.md) — Allowed non-model utilities (`Tesseract`, `pypdfium2`, `pdfplumber`, `OpenCV`, `pypdf`, `python-docx`)
- [schema/medical_extraction.schema.json](schema/medical_extraction.schema.json) — Pydantic v2 JSON Schema contract
- [PROJECT_STATUS.md](PROJECT_STATUS.md) — Honest phase & capability status
- [DECISIONS/](DECISIONS/) — Architecture Decision Records (`ADR-0001` to `ADR-0008`)
- [extras/](extras/) — Frozen non-core modules isolated per `ADR-0008`
