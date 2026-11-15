# Realistic & Out-of-Distribution Clinical Evaluation Protocol

- **Version:** 1.1 (`SYNTH-MED-V3.1` & External Clinical Validation)
- **Date:** 2026-11-15
- **Scope:** Bilingual (`en`, `fr`) clinical NER, assertion classification, relation extraction, TIMEX normalization, document classification, and OCR stress evaluation.

---

## 1. Purpose & Motivation

High in-distribution (IID) F1 on synthetic notes does **not** prove clinical generalization. Earlier iterations (`SYNTH-MED-V2`) exhibited 94.8% masked sentence overlap and 100% clinical entity overlap between train and test splits. This protocol defines strict separation rules, zero-leakage auditing, and realistic degradation testing so reported metrics measure true lexical, structural, and document-imaging generalization.

---

## 2. Partitioning & Anti-Leakage Design (`SYNTH-MED-V3.1`)

### 2.1 Four Disjoint Evaluation Regimes
1. **`val` (160 docs: 80 EN + 80 FR):** Used exclusively for hyperparameter tuning, early stopping, and Stage 1–5 ablation selection.
2. **`test_iid` (160 docs: 80 EN + 80 FR):** Held-out IID sample from the training lexicon/template distribution.
3. **`test_unseen_terms` (160 docs: 80 EN + 80 FR):** Uses a **disjoint 30% clinical concept partition** (`lexicon_v3.py`, partitioned at concept/ICD/ATC/LOINC level before surface-variant expansion). Target: **0.0% clinical-term overlap** with `train`.
4. **`test_unseen_templates` (160 docs: 80 EN + 80 FR):** Uses a **disjoint 30% sentence template partition** (`templates_v3.py`, partitioned per `(language, section)`). Target: **< 5% digit-masked 5-gram sentence overlap** with `train`.

### 2.2 Document-Type Structural Signal (V3.1)
Each of the 6 document types (`discharge_summary`, `prescription`, `lab_report`, `radiology`, `operative`, `consultation`) is generated with a dedicated header pool (12% intentionally ambiguous headers), dedicated section set, and realistic section ordering. Approaches **must never** inspect `doc.doc_type` at prediction time (verified by `tests/test_hidden_doc_type_and_learnability.py`).

### 2.3 Mandatory Corpus Audit
After any generator or lexicon change, `scripts/audit_corpus.py` must verify:
- `0` doubled span texts or adjacent duplicate n-grams;
- `100.0%` character-offset and relation-endpoint integrity;
- `0.0%` clinical-term overlap in `test_unseen_terms`;
- `< 5%` masked sentence overlap in `test_unseen_templates`.

---

## 3. Realistic OCR & Imaging Stress Protocol

To measure document-processing robustness beyond clean text:
- **Sample size:** `>= 60` pages per language (`120` pages total: 60 EN + 60 FR from `test_iid`).
- **Degradation dimensions:** Resolution (`300 / 200 / 150` DPI), JPEG compression (`35–90`), Gaussian blur, additive sensor noise, rotation (`±5°`), perspective keystone warp (mobile phone capture), uneven illumination gradients, stamps/stains, two-column layouts, bordered tables, and `>= 3` distinct font families.
- **End-to-end scoring:** Measure CER/WER per tier and language **and** end-to-end extraction strict F1 (best single model + hybrid) on OCR text mapped back to reference character offsets (`map_ocr_spans_to_reference`), reporting exact F1 drop (`ocr_f1 - clean_f1`).

---

## 4. Protocol for Future Real / External Clinical Corpora

Before evaluating on any real institutional or public clinical corpus (e.g. de-identified hospital EHR notes, i2b2/n2c2, QUAERO French Medical):
1. **Governance & Privacy:** Verify IRB/ethics approval or public DUA; run `medresearch_platform/security/scrubber.py` and never store raw PHI/PII in git or logs.
2. **Annotation Alignment:** Map external entity/assertion/TIMEX labels to the canonical contract (`schema/medical_extraction.schema.json`) without modifying locked synthetic test sets.
3. **Locked Test Discipline:** Record every locked test evaluation in `docs/results/TEST_SET_LOG.md` with dataset, split, commit hash, seed list, and reason.
4. **Statistical Reporting:** Evaluate `5` random seeds per final config, compute `>= 10,000`-resample cluster-paired document bootstrap tests pooled across seeds, and apply Holm-Bonferroni correction across all pairwise comparisons in each table.
