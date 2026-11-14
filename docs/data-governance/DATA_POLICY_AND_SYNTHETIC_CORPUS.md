# Data Governance Policy & Synthetic Bilingual Corpus (`SYNTH-MED-EN-FR v1.0.0`)

**Specification References:** Sections 2, 15, 16, 17, 18, 25

## 1. Privacy-Safe Development Data Policy
- **Zero Unauthorized PHI:** No real patient records or unauthorized Protected Health Information (PHI) are stored in the repository, test fixtures, logs, or experiment artifacts.
- **Synthetic Corpus Identity:**
  - **Dataset ID:** `ds_synth_bilingual_v1` (`SYNTH-MED-EN-FR`, version `1.0.0`)
  - **Source Type:** `SYNTHETIC` (`SYNTHETIC_NO_PHI`)
  - **License:** `CC0-1.0-SYNTHETIC-RESEARCH-ONLY`
  - **Manifest SHA-256:** `f637bc1496a6a4b423e264b43e7727875029c7d4f7081fae291eb3ff87a28539`
  - **Languages:** English (`en`), French (`fr`), and Mixed English/French (`mixed_en_fr`)
  - **Modalities / File Formats:** Native `TXT`, `PDF`, `DOCX`, and synthetic scanned `PNG` glyph blocks.

## 2. Patient & Document-Family Leak-Free Splitting
All records in `SYNTH-MED-EN-FR` are partitioned strictly by `subject_key` and `family_key`:
- **Train Split (`train`):** `subj_synth_001` (`fam_synth_001_a`, `fam_synth_001_b` — longitudinal admission & follow-up notes in English)
- **Validation Split (`val`):** `subj_synth_002` (`fam_synth_002_a` — French hospital summary DOCX)
- **Locked Test Split (`locked_test`):** `subj_synth_003` (`fam_synth_003_a` — Mixed EN/FR consultation note)

Automated verification (`verify_leak_free_splits`) confirms **0 overlapping subjects** and **0 overlapping document families** across `train`, `val`, and `locked_test`.
