# Independent Human Clinical Evaluation Protocol (EN / FR / Code-Switched)

> **Status:** Protocol specification ready for execution. **No human-reviewed clinical conversation gold set exists yet**; synthetic benchmarks (`SYNTH-MED-V3.1`, `medical_conversation_dataset_en_fr.zip`) must never be presented as human-verified real-world clinical validation.

---

## 1. Purpose & Strict Isolation Rules
- Establish an **independent, human-annotated gold-standard evaluation set** to measure real-world generalization of the existing from-scratch models (`HybridPipeline`, `ClassicalCRFModel`, `PyTorchCharCNNBiLSTMCRF`).
- **Strict isolation:** Documents in this set MUST NEVER be used for training, vocabulary construction, feature engineering, hyperparameter tuning, or calibration fitting.
- **Privacy & Governance:** Use only IRB/ethics-approved de-identified clinical transcripts (HIPAA Safe Harbor / GDPR / CNIL compliant) or newly clinician-authored realistic dialogues with zero real PHI. Never scrape private medical records.

## 2. Stratified Sampling Design (Target Minimum: `n = 150` Documents)
- **Languages:** `65` English (`en`), `65` French (`fr`), `20` bilingual code-switched (`mixed`).
- **Specialties:** Stratified across General/Internal Medicine, Emergency Medicine, Cardiology, Neurology, Pulmonology, Endocrinology, Pediatrics, Oncology, Dermatology, and Radiology.
- **Noise & Artifact Coverage:** Include clean dictated transcripts, realistic ASR transcripts (hesitations, repetitions, missing punctuation), scanned/OCR-corrupted notes, and documents containing genuine clinical contradictions.

## 3. Orthogonal 3-Axis Annotation Guidelines
Annotators label every clinical entity span (`char_start`, `char_end`, exact substring) independently across **three orthogonal axes** (never conflating them):
1. **Assertion:** `present` | `absent` | `possible` | `conditional` | `hypothetical` | `conflicting` (missing mention ≠ `absent`).
2. **Temporality:** `current` | `historical` | `planned` | `resolved` | `recurrent` | `unknown`.
3. **Experiencer:** `patient` | `family` (with relationship: `mother`/`father`/`brother`/`sister`/`other_family`) | `caregiver` | `other`.

Additional required annotations:
- **Relations:** `HAS_DOSAGE`, `HAS_ROUTE`, `HAS_FREQUENCY`, `TREATS`, `HAS_VALUE`, `HAS_UNIT`, `LOCATED_IN`, `REVEALS`, with explicit directionality and evidence spans.
- **Medications:** Distinguish `prescribed`, `ordered_not_administered`, `administered`, `patient_reported_active`, `not_taking`, `discontinued`, `historical`.
- **Contradictions:** Preserve both sides with turn/section provenance; never silently resolve.

## 4. Double-Annotation & Inter-Annotator Agreement (IAA)
- **Minimum 2 independent clinical annotators** on at least `30%` of the corpus (`n >= 45` documents), with a 3rd senior clinician adjudicating disagreements.
- **Required IAA Metrics:**
  - Entity spans: pairwise strict & lenient span F1 (target `>= 0.85`).
  - Assertion, Temporality, Experiencer: Cohen's $\kappa$ and Krippendorff's $\alpha$ (target $\kappa \ge 0.80$).
  - Relations & Contradictions: pairwise F1 (target `>= 0.80`).
- **Promotion Gate for Provisional Drafts:** `tests/fixtures/noise_0027_ref_draft.json` and `noise_0038_ref_draft.json` remain `"status": "draft_unreviewed"` until adjudicated under this protocol with `reviewer_signoff` and `reviewed_at` recorded.
