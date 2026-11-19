# Open-License Clinical Corpora Card (`open_clinical`)

**Built / Verified:** 2026-11-19  
**Builder Script:** [`scripts/download_open_clinical_corpora.py`](../../scripts/download_open_clinical_corpora.py)  
**Converters:** [`medresearch_platform/data/converters.py`](../../medresearch_platform/data/converters.py) (`from_maccrobat_brat`, `from_pxcorpus_row`, `from_e3c_xmi`)  
**Output Directory (gitignored):** `data/processed/open_clinical/`

---

## 1. Included Real Clinical Corpora & Licenses

| Corpus | Language(s) | Documents | License | Source / DOI | Clinical Coverage |
|---|---|---:|---|---|---|
| **MACCROBAT2020** | English (`en`) | 200 | **`CC-BY-4.0`** | Figshare DOI `10.6084/m9.figshare.9764942` | Real PubMed Central clinical case reports (BRAT `.txt`/`.ann`): 11 canonical entity types (`DIAGNOSIS`, `SYMPTOM`, `MEDICATION`, `DOSAGE`, `FREQUENCY`, `ROUTE`, `DURATION`, `PROCEDURE`, `LAB_VALUE`, `ANATOMY`, `DATE`) |
| **PxCorpus (PxSLU)** | French (`fr`) | 1,981 | **`CC-BY-4.0`** | Zenodo DOI `10.5281/zenodo.6482586` · HuggingFace `DrBenchmark/PxCorpus` | Real French drug prescriptions (`train`: 1,386, `val`: 198, `test`: 397): `MEDICATION`, `DOSAGE`, `DURATION`, `FREQUENCY`, `ROUTE` |
| **E3C Layer 1** | English (`84`) + French (`81`) | 165 | **`CC-BY-NC-4.0`** (annotations); source journals: `158` `CC BY`/`CC BY 4.0` (incl. *The Pan African Medical Journal*) + `7` `CC BY-NC` | GitHub `hltfbk/E3C-Corpus` | Manually annotated EN + FR clinical narratives (UIMA XMI): `DIAGNOSIS`, `ANATOMY`, `LAB_VALUE`, `DATE`/`DURATION`/`FREQUENCY` (`TIMEX3` ISO values), and `EVENT` polarity/modality (`absent`, `possible`, `hypothetical`) |

> **Permissive-Only Mode:** Pass `--permissive-only` to `scripts/download_open_clinical_corpora.py` to exclude `E3C` (`CC-BY-NC-4.0`) and retain strictly `CC-BY-4.0` documents (`MACCROBAT2020` + `PxCorpus` = `2,181` documents).

---

## 2. Verified Dataset Statistics (`data/processed/open_clinical/MANIFEST.json`)

- **Total Documents:** `2,346` (`284` EN + `2,062` FR)
- **Document-Level Splits (zero cross-split leakage):**
  - `train`: `1,645`
  - `val`: `258`
  - `test`: `443`
- **Total Validated Spans:** `30,962` entities (`100%` passing `AnnotatedDocument.validate_spans()`) + `4,952` TIMEX spans.
- **Entity Distribution:**
  - `PROCEDURE`: `5,619` | `ANATOMY`: `4,382` | `LAB_VALUE`: `3,832` | `DIAGNOSIS`: `3,539` | `SYMPTOM`: `3,368` | `MEDICATION`: `2,547` | `DOSAGE`: `2,496` | `DURATION`: `2,051` | `FREQUENCY`: `1,979` | `DATE`: `921` | `ROUTE`: `228`

---

## 3. Reproducible Build Command

```bash
# Full research set (MACCROBAT2020 + PxCorpus + E3C EN/FR)
python3 scripts/download_open_clinical_corpora.py

# Permissive CC-BY-4.0 only (MACCROBAT2020 + PxCorpus)
python3 scripts/download_open_clinical_corpora.py --permissive-only
```
