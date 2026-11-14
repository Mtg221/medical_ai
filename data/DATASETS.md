# Dataset Catalog, Licences, and Provenance

**Policy:**
1. **No real patient data is committed to Git.** `data/raw/` and `data/processed/` are git-ignored.
2. Any dataset requiring a Data Use Agreement (DUA) or institutional licence is **never** downloaded automatically; explicit user confirmation and recorded DUA reference are required.
3. **Synthetic results and real-data results are always reported separately.**

---

## 1. Evaluated Public / External Datasets

| Dataset | Language | Tasks Covered | Licence / Access Requirement | Status | Label Mapping Notes |
|---|---|---|---|---|---|
| **Synthetic Bilingual Clinical Corpus v2 (`SYNTH-MED-V2`)** | EN, FR | NER (14 types), Assertion (6), Temporality (3), Relations (9), TIMEX, Doc Type (6) | Licence-free (generated locally by `medresearch_platform.data.synthetic_v2`) | **Active (Phase 1)** | Gold spans & relations by construction |
| **QUAERO French Medical Corpus** | FR | Medical entities (UMLS semantic groups) | Open for research (ELRA / LIMSI) | Evaluated — converter ready | Maps `DISO`→`DIAGNOSIS`/`SYMPTOM`, `CHEM`→`MEDICATION`, `PROC`→`PROCEDURE`, `ANAT`→`ANATOMY`; other UMLS groups logged as unmapped |
| **E3C (European Clinical Case Corpus)** | EN, FR | Clinical entities, temporal expressions (TIMEX3) | CC BY-NC 4.0 (research) | Evaluated — converter ready | Maps clinical entity & TIMEX layers; non-clinical metadata unmapped |
| **DEFT / CAS French Clinical Cases** | FR | Clinical cases, negation, uncertainty, entities | Research agreement required | **Requires agreement** (ask user before download) | — |
| **FRASIMED** | FR | Clinical NER | Open research corpus (check repo licence) | Candidate | — |
| **n2c2 / i2b2 2010, 2012, 2018** | EN | Concepts, assertions, relations, temporal, medications | **DUA required** (DBMI Portal) | **Requires DUA** (ask user before download) | Converter maps `problem`→`DIAGNOSIS`/`SYMPTOM`, `treatment`→`MEDICATION`/`PROCEDURE`, `test`→`LAB_TEST`, assertion classes |
| **BC5CDR / NCBI-Disease / MedMentions** | EN | Biomedical abstracts (diseases, chemicals) | Open (PubMed / BioCreative) | Candidate (biomedical abstracts, not clinical notes) | Maps `Disease`→`DIAGNOSIS`, `Chemical`→`MEDICATION` |
| **MTSamples** | EN | Transcribed clinical notes + medical specialty | Public web archive (check terms before redistribution) | Candidate for doc-type classification | Specialty → `doc_type` mapping |

---

## 2. Format Support & Annotation Tool Path

- **BRAT Standoff (`.txt` + `.ann`)**: Full import/export for `T*` entities, `R*` relations, and `A*` attributes (`Assertion`, `Temporality`).
- **Label Studio JSON**: Full import/export for span annotations and document classification labels.
- **Canonical Internal Format (`JSONL`)**: `AnnotatedDocument` schema with verified character offsets (`text[start:end] == span.text`), split assignment (`train` / `val` / `test`), and SHA-256 hash manifest.
