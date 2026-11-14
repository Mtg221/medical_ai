# Bilingual Medical Annotation Guidelines (EN / FR)

**Version:** `1.0`  
**Languages:** English (`en`), French (`fr`), extensible to additional languages via configuration.

---

## 1. Entity Types (14 Types)

| Entity Type | Definition | English Example | French Example |
|---|---|---|---|
| `SYMPTOM` | Patient-reported or observed clinical sign/symptom | `shortness of breath`, `chest pain` | `dyspnée`, `douleur thoracique` |
| `DIAGNOSIS` | Confirmed, suspected, ruled-out, or historical condition/disease | `type 2 diabetes mellitus`, `pulmonary embolism` | `diabète de type 2`, `embolie pulmonaire` |
| `MEDICATION` | Drug or therapeutic substance name (generic or brand) | `metformin`, `amoxicillin` | `metformine`, `amoxicilline` |
| `DOSAGE` | Dose strength or quantity | `500 mg`, `20 mg` | `500 mg`, `1000 mg` |
| `FREQUENCY` | Administration schedule/frequency | `twice daily`, `once daily` | `deux fois par jour`, `1 fois/jour` |
| `ROUTE` | Route of administration | `orally`, `intravenous` | `par voie orale`, `IV` |
| `DURATION` | Treatment or symptom duration | `for 7 days`, `3 days` | `pendant 7 jours`, `depuis 3 jours` |
| `LAB_TEST` | Laboratory or physiological measurement name | `HbA1c`, `CRP`, `serum creatinine` | `HbA1c`, `CRP`, `créatininémie` |
| `LAB_VALUE` | Numeric or qualitative test result | `7.4`, `18.2` | `7,4`, `18,2` |
| `LAB_UNIT` | Measurement unit | `%`, `mg/L`, `µmol/L` | `%`, `mg/L`, `µmol/L` |
| `PROCEDURE` | Diagnostic, imaging, or surgical procedure | `chest CT`, `echocardiography` | `angioscanner thoracique`, `échocardiographie` |
| `ANATOMY` | Anatomical site or organ | `lungs`, `left lower lobe` | `poumons`, `lobe inférieur gauche` |
| `DATE` | Calendar or relative date expression | `2026-03-10`, `3 days ago` | `10/03/2026`, `il y a 3 jours` |
| `PATIENT_ID` | Synthetic/pseudonymized patient identifier | `SYNTH-1001` | `SYNTH-2042` |

**Span boundary rules:**
- Annotate the **minimal complete clinical phrase** without trailing punctuation (e.g., `chest pain`, not `chest pain.`).
- Keep `MEDICATION`, `DOSAGE`, `FREQUENCY`, and `ROUTE` as separate spans linked by relations.
- Keep `LAB_TEST`, `LAB_VALUE`, and `LAB_UNIT` as separate spans linked by `HAS_VALUE` / `HAS_UNIT`.

---

## 2. Assertion & Temporality Attributes

Every entity carries:
- **`assertion`** (6 classes):
  - `present` — affirmed in the patient (e.g., *"Presents with fever"*, *"Présente une fièvre"*)
  - `absent` — explicitly negated/ruled out (e.g., *"pulmonary embolism ruled out"*, *"absence d'embolie pulmonaire"*)
  - `possible` — uncertain/suspected/differential (e.g., *"suspected bacterial pneumonia"*, *"pneumopathie suspectée"*)
  - `hypothetical` — conditional/contingent (e.g., *"return if fever develops"*, *"consulter si apparition de fièvre"*)
  - `family` — family history (e.g., *"Family history of atrial fibrillation"*, *"Antécédents familiaux de fibrillation auriculaire"*)
  - `historical` — past medical history, not active acute complaint (e.g., *"History of essential hypertension"*, *"Antécédent d'hypertension artérielle"*)
- **`temporality`** (3 classes):
  - `current` | `past` | `planned`

---

## 3. Relation Types

| Relation | Head → Tail | Example |
|---|---|---|
| `TREATS` | `MEDICATION` → `DIAGNOSIS` / `SYMPTOM` | `amoxicillin` → `bacterial pneumonia` |
| `HAS_DOSAGE` | `MEDICATION` → `DOSAGE` | `metformin` → `500 mg` |
| `HAS_FREQUENCY` | `MEDICATION` → `FREQUENCY` | `metformin` → `twice daily` |
| `HAS_ROUTE` | `MEDICATION` → `ROUTE` | `metformin` → `orally` |
| `HAS_VALUE` | `LAB_TEST` → `LAB_VALUE` | `HbA1c` → `7.4` |
| `HAS_UNIT` | `LAB_VALUE` → `LAB_UNIT` | `7.4` → `%` |
| `REVEALS` | `PROCEDURE` / `LAB_TEST` → `DIAGNOSIS` | `chest CT` → `pulmonary embolism` |
| `CAUSES` | `DIAGNOSIS` / `MEDICATION` → `SYMPTOM` | `bacterial pneumonia` → `dyspnea` |
| `LOCATED_IN` | `SYMPTOM` / `DIAGNOSIS` / `PROCEDURE` → `ANATOMY` | `bacterial pneumonia` → `lungs` |

---

## 4. TIMEX Types

| TIMEX Type | Definition | Examples (EN / FR) |
|---|---|---|
| `DATE` | Calendar date | `2026-03-10`, `10/03/2026` |
| `DURATION` | Span of time | `for 7 days`, `pendant 7 jours` |
| `FREQUENCY` | Recurrence | `twice daily`, `deux fois par jour` |
| `RELATIVE` | Relative to document date | `3 days ago`, `il y a 3 jours` |

---

## 5. Document Types (6 Classes)

`discharge_summary`, `consultation`, `lab_report`, `prescription`, `radiology_report`, `operative_note`.
