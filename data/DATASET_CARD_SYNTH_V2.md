# Dataset Card: Synthetic Bilingual Clinical Corpus v2 (`SYNTH-MED-V2`)

- **Version:** `1.0.0` (Config: `configs/data/v1.yaml`)
- **Languages:** English (`en`) and French (`fr`)
- **Licence:** Licence-free, generated deterministically on local CPU (`seed=42`)
- **Intended Use:** Development, unit/integration testing, and controlled CPU-only benchmarking of all 6 extraction approaches (NER, assertion, temporality, relations, TIMEX, document classification, and digital-vs-scanned OCR robustness).
- **Explicit Limitations:**
  - Template/grammar-generated synthetic clinical notes — **does not** capture the full lexical, stylistic, and abbreviation diversity of real hospital EHRs.
  - Results on `SYNTH-MED-V2` **must** be reported as *synthetic benchmark results* and never conflated with real clinical validation.

---

## Composition & Splits

- **Target Volume:** 800 documents (`400 EN + 400 FR`) across 6 document types (`discharge_summary`, `consultation`, `lab_report`, `prescription`, `radiology_report`, `operative_note`).
- **Split Policy:** 70% `train` / 15% `val` / 15% `test` (patient-level and document-level, stratified by `language` and `doc_type`).
- **Leakage Controls:**
  - Zero shared `patient_id` across splits.
  - MinHash 5-gram shingle Jaccard similarity check across splits (threshold `0.85`).
  - `test` split is locked and verified via SHA-256 manifest.
