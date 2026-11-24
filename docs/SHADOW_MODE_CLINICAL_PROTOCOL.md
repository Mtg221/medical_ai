# Prospective Shadow-Mode Clinical Evaluation Protocol (Phase 4)

**Status:** Protocol & Automated Gate Harness Ready (`scripts/eval_shadow_mode.py`)  
**Scope:** Supervised hospital shadow-mode validation for bilingual EN/FR Medical NLP and Dermatology Vision Triage prior to any clinical pilot or regulatory submission.

---

## 1. Objective & Clinical Safety Boundary

Before any deployment beyond research prototyping, the platform must run in **read-only shadow mode** alongside standard clinical care:
- AI predictions are logged silently and **never** shown to patients or used to alter clinical decisions during shadow mode.
- Ground truth is established independently via histopathology biopsy (dermatology) or double-blinded clinician adjudication (NLP clinical notes).

## 2. Mandatory Promotion Gates (`scripts/eval_shadow_mode.py`)

All 8 quantitative safety, calibration, fairness, and latency gates must pass simultaneously before promoting from shadow mode to a supervised clinician-in-the-loop pilot:

| Gate | Threshold | Rationale |
|---|---|---|
| Dermoscopic AUROC | `>= 0.94` | Maintain high triage discrimination on dermoscopy |
| Dermoscopic Sensitivity @ 90% Specificity | `>= 0.90` | Minimize missed malignancies |
| Clinical Smartphone AUROC | `>= 0.88` | External smartphone generalization |
| Clinical Smartphone Sensitivity @ 90% Specificity | `>= 0.75` | Upgrade required via PAD-UFES-20 / Fitzpatrick17k / DDI before clinical-photo scoring |
| Overall Calibration (ECE) | `<= 0.05` | Reliable probability estimates for clinician decision support |
| Fitzpatrick Subgroup Sensitivity Gap (`I-II` vs `III-IV` vs `V-VI`) | `<= 0.08` | Equity across all skin phototypes |
| Serving Latency `p95` | `<= 2500 ms` | Practical clinical workflow responsiveness |
| Clinician Agreement Rate | `>= 0.85` | Human-factors alignment on non-abstained outputs |

## 3. Running the Evaluation Harness

```bash
# 1. Ingest external clinical smartphone datasets with zero-leakage SHA-256 deduplication
python scripts/ingest_external_clinical_derm.py \
  --csv /path/to/pad_ufes_20/metadata.csv \
  --images-dir /path/to/pad_ufes_20/images \
  --dataset pad_ufes_20 \
  --out var/cache/derm/meta/manifest-pad_ufes_20.json

# 2. Evaluate prospective shadow-mode JSONL log against promotion gates
python scripts/eval_shadow_mode.py \
  --input /path/to/shadow_mode_records.jsonl \
  --out docs/results/shadow_mode_eval.json
```

## 4. Expected Shadow-Mode JSONL Record Schema

Each line in `shadow_mode_records.jsonl` must contain de-identified records only (zero PHI):
```json
{
  "case_id": "SHD-0001",
  "modality": "dermoscopic",
  "fitzpatrick": "V-VI",
  "predicted_prob_malignant": 0.88,
  "abstained": false,
  "ground_truth_malignant": 1,
  "clinician_agreed": true,
  "latency_ms": 620.0
}
```
