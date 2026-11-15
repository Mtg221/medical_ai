# Multilingual Medical Information Extraction AI (`medical_ai`)

**Version:** `0.2.0` (Phases 0–12 + Correction Prompts 11–14)  
**Primary Languages:** English (`en`) & French (`fr`)  
**Execution Profile:** CPU-Only Native (`Intel Core i7-9750H`, 32 GB RAM, macOS Darwin `x86_64`)  
**Primary AI Track:** Models trained strictly from scratch (random initialization, zero pretrained weights)

> **Clinical & Scientific Safety Notice**  
> This platform is a **research tool — not for diagnosis or treatment; human review is required before any clinical use**. Do not use AI-extracted or inferred outputs as verified clinical truth.

---

## 1. Architecture & Verified Results (`SYNTH-MED-V3.1` Locked Test Splits, 5 Seeds `1..5`)

All primary models are trained from scratch on `train` (600 docs: 300 EN + 300 FR) and evaluated on `val` (160) and three locked test splits (`test_iid` 160, `test_unseen_terms` 160, `test_unseen_templates` 160). Full methodology, ablations, OCR benchmarks, and limitations are documented in [docs/RESEARCH_REPORT.md](docs/RESEARCH_REPORT.md).

| Approach | `test_iid` Strict F1 [Doc Boot 95% CI] | `test_unseen_terms` Strict F1 | `test_unseen_templates` Strict F1 | `test_iid` Rel F1 | `test_iid` TIMEX F1 |
|---|---|---|---|---|---|
| 1. `rule_based` | `0.3221` [0.2964, 0.3476] | `0.3233` | `0.3066` | `0.0425` | `0.5101` |
| 2. `classical_ml` (TF-IDF + LinearSVC) | **`0.9983`** [0.9966, 0.9997] | `0.8441` | **`0.9670`** | `0.7340` | `0.7160` |
| 3. `classical_crf` (`sklearn-crfsuite`) | `0.9968` [0.9948, 0.9986] | `0.9169` | `0.9527` | `0.7341` | `0.7167` |
| 4. `bilstm` (from-scratch NumPy) | `0.9659` [0.9613, 0.9703] | `0.7859` | `0.8619` | `0.7208` | `0.6997` |
| 5. `bilstm_crf` (from-scratch NumPy) | `0.9884` [0.9859, 0.9907] | `0.8141` | `0.9392` | `0.7342` | `0.7155` |
| 6. `transformer` (from-scratch NumPy) | `0.7930` [0.7764, 0.8100] | `0.5241` | `0.6886` | `0.5978` | `0.4557` |
| 7. `hybrid` (V3.1 5-stage) | `0.9941` [0.9920, 0.9959] | `0.8277` | `0.8949` | **`0.8922`** | **`0.9986`** |
| **8. `hybrid_v2` (`val`-selected, served)** | `0.9972` [0.9954, 0.9987] | **`0.9204`** | `0.9310` | `0.8912` | **`0.9986`** |

---

## 2. Quick Start

```bash
# 1. Install package (editable) and dev tools
pip install -e ".[dev]"

# 2. Build React + TypeScript + Tailwind frontend (served at /react)
npm --prefix frontend ci && npm --prefix frontend run build

# 3. Build dataset splits, train & save the served Hybrid v2 model (~138s on CPU), and run experiments
medextract data build --config configs/data/v3_1.yaml
medextract train --config configs/exp/v3_2/hybrid_v2.yaml --seed 1 --out models/hybrid_v2_seed1/
medextract experiment run configs/exp/hybrid.yaml --seeds 1,2,3

# 4. Extract structured JSON or EN/FR report from a document using the saved model (TXT, PDF, DOCX, PNG)
medextract extract path/to/note.pdf --model models/hybrid_v2_seed1 --out extracted.json
medextract report path/to/note.pdf --model models/hybrid_v2_seed1 --lang fr --format html --out report_fr.html

# 5. Initialize local demo users and start the FastAPI server
# macOS / Linux:
export MEDRESEARCH_DEMO_PASSWORD="ChangeMe_LocalDev_2026!"
medextract users init
uvicorn medresearch_platform.interfaces.api:create_app --factory --host 127.0.0.1 --port 8000

# Windows (PowerShell):
# $env:MEDRESEARCH_DEMO_PASSWORD = "ChangeMe_LocalDev_2026!"
# medextract users init
# uvicorn medresearch_platform.interfaces.api:create_app --factory --host 127.0.0.1 --port 8000

# Reset a specific user's password at any time:
medextract users set-password admin_dakar --password "ChangeMe_LocalDev_2026!"

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
