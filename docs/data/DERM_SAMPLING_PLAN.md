# Dermatology Corpus Sampling Plan (`derm-corpus` v2)

> **Status:** Approved by: owner (guest777) on 2026-11-18 (recorded after the fact, via in-session answer "Yes, approve the v2 plan as written"). The v2 bulk sync itself ran earlier, before this approval was recorded (W13). `locked_test_v2` is superseded pending the `locked_test_v3` plan, which needs its own separate written approval.  
> **Target Hardware:** Intel Core i7-9750H, 32 GB RAM, CPU-only.  
> **Rules:** Real images only (`ok == true`, research-allowed licence only), lesion-grouped splits (`mk` group key), zero fabricated labels (`mg` unknown/indeterminate masked), cross-split dHash deduplication (drop near-duplicates from test, not train).

---

## 1. Corpus Inventory (All 14 Manifest Pages)

Scanned across all 14 manifest pages in `var/cache/derm/meta/` (`manifest-clinical-000.json`, `manifest-dermoscopic-000..002.json`, `manifest-tbp_tile-000..008.json`):

- **Total records in manifests:** `573,024`
- **Eligible (`quality_ok == true` AND `research_allowed == true`):** `305,964`

| Dimension | Eligible Breakdown (`n = 305,964`) |
|---|---|
| **Modality** | `tbp_tile`: `234,467` · `dermoscopic`: `65,758` · `clinical`: `5,739` |
| **Dataset** | `isic`: `304,254` · `skindisnet`: `1,710` |
| **Malignancy (`mg`)** | `benign`: `277,615` · `malignant`: `7,786` · `unknown`: `18,749` · `indeterminate`: `1,814` |
| **Fitzpatrick Tone** | `I`: `1,116` · `II`: `3,324` · `III`: `1,663` · `IV`: `922` · `V`: `823` · `VI`: `751` · `UNKNOWN`: `297,365` |
| **Confirmation (`cf`)** | `histopathology`: `17,303` · `single contributor clinical assessment`: `33,480` · `single image expert consensus`: `25,400` · `serial imaging showing no change`: `10,500` · `None/other`: `219,281` |

---

## 2. Target Split & Stratification Design

- **Total target thumbnails:** `~30,000` (96–128 px thumbnails)
  - **`train`:** `24,000` thumbnails (target range `20,000–30,000`)
  - **`val`:** `3,000` thumbnails
  - **`locked_test_v2`:** `3,000` thumbnails
- **Grouping & Leak Prevention:**
  - Every record is assigned to `train` / `val` / `locked_test_v2` strictly by its **lesion group key** (`mk` when present, fallback `dataset:image_id` in `catalog.py`). Zero lesion keys may appear in more than one split (enforced by unit test).
- **Stratification & Oversampling Quotas:**
  1. **Fitzpatrick skin tones (`I`–`VI`):** Oversample all `8,599` eligible tone-labelled images (`I=1,116`, `II=3,324`, `III=1,663`, `IV=922`, `V=823`, `VI=751`) across lesion groups so `locked_test_v2` has **≥ 150 images per tone `I`–`VI`** wherever available, while keeping ≥ 70% of each tone in `train` + `val`. If any `tone × malignant` cell has fewer images in the corpus, report the exact real count (never pad or duplicate).
  2. **Modalities:** Ensure `locked_test_v2` has **≥ 300 images per modality** (`clinical`, `dermoscopic`, `tbp_tile`) and `val` has ≥ 300 per modality, with the remainder in `train`.
  3. **Malignancy (`mg`) & Dataset (`cf`):** Oversample `malignant` lesions (`7,786` eligible) so `train`, `val`, and `locked_test_v2` each contain substantial malignant representation across biopsy-confirmed (`histopathology`) and clinical/atlas subsets. Records with `mg` in `("unknown", "indeterminate")` are **masked** (excluded from malignancy loss and malignancy metrics, never labelled `"benign"`).
  4. **Stratification cell:** `dataset × modality × malignancy_group × diagnosis_harmonised × skin_tone`.

---

## 3. Cross-Split Perceptual Deduplication & Frozen Test List

1. After downloading the planned thumbnails into `var/cache/derm/thumbs/`, compute 64-bit dHash (`compute_dhash64_from_path`) on every downloaded image.
2. Scan `locked_test_v2` against `train` and `val` (hamming distance `<= 4`):
   - **Drop near-duplicates from `locked_test_v2`**, never from `train`.
3. Write the final frozen test image IDs (one per line) to `configs/vision/locked_test_v2.txt` and record its SHA-256 in `docs/results/TEST_SET_LOG.md` **before** any model training.

---

## 4. Estimated Download Size, Disk Space & Time

- **Measured average thumbnail size** (from `671` cached thumbnails in `var/cache/derm/thumbs/`): **`15.57 KB`**
- **Estimated download payload (`30,000` thumbnails):** `~467 MB` (`~450–550 MB`)
- **Disk space required:** `~650 MB` for thumbnails + `~315 MB` existing manifests = `~1.0 GB` total (recommend **`≥ 2 GB` free disk space**).
- **Estimated download time:** **`~25–45 minutes`** at ~12–20 thumbnails/sec (resumable, idempotent; already-cached thumbnails are skipped).

---

## 5. Execution Command (Run by Owner)

Credentials stay in `.env` (`DERM_CLOUDFRONT_USER`, `DERM_CLOUDFRONT_PASS`) and are never printed or committed:

```bash
medextract derm sync --plan docs/data/DERM_SAMPLING_PLAN.md
```

After the sync finishes, generate the frozen test list and data card (`counts per split × tone × modality × mg × dataset`) for inspection **before** training any router or CNN model.
