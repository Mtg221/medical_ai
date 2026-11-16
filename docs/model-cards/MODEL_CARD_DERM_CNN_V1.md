# Model Card: `DermResNetMultiTask-v1.0.0` & `PictureTriageRouterCNN-v1.0.0`

- **Model Family:** From-Scratch 4-Stage Residual CNN (`16 → 32 → 64 → 128` channels, `130,590` parameters, `1.04 MB`) + 5-Class Picture Triage Router (`1,829` parameters)
- **Initialization:** 100% Random Initialization (`he_normal` + `xavier_uniform`, `pretrained_weights_used = false`, zero ImageNet/external weights)
- **Execution Profile:** CPU-Only Native (`macOS x86_64`, Linux, Windows; optional `.[vision]` PyTorch extra documented in `ADR-0010`)
- **Artefact Locations:** `models/derm_cnn_seed1/` (`manifest.json` + `weights.npz`), `models/vision_router_v1/`

> **Clinical & Scientific Safety Notice (EN / FR)**
> - **EN:** Research tool — not a diagnosis. A dermatologist must examine the lesion.
> - **FR :** Outil de recherche — ne constitue pas un diagnostic. Un dermatologue doit examiner la lésion.

---

## 1. Intended Use & Out-of-Scope / Not-Intended Use

- **Intended Use:**
  - Research benchmarking of from-scratch 2D dermatology risk estimation, top-3 harmonised diagnosis ranking, picture modality triage (`skin_clinical`, `skin_dermoscopic`, `skin_tbp_tile`, `document_scan`, `other`), bilingual image-quality gating, and spatial Grad-CAM explanations on `derm-corpus` splits.
  - Human-in-the-loop research review inside an authenticated institutional workspace (`review_status = "needs_review"` on high-risk or uncertain outputs).
- **Not-Intended / Prohibited Use:**
  - **NOT** a medical device and **NOT** cleared or validated for autonomous clinical diagnosis, triage, biopsy deferral, or treatment decisions.
  - Must **never** be deployed without mandatory dermatologist examination of the lesion.

---

## 2. Training Data, Splits & Licences (`derm-corpus`)

- **Metadata Catalog:** `573,024` records (`isic`: `549,541`, `fitzpatrick17k`: `11,394`, `scin`: `10,379`, `skindisnet`: `1,710`), detailed in [DERM_CORPUS_CARD.md](../data/DERM_CORPUS_CARD.md).
- **Stratified Evaluation Subset:** `871` cached thumbnails (`train`: `613`, `val`: `140`, `locked_test`: `118`) partitioned deterministically by lesion/image group key with `0` cross-split overlap and 64-bit `dHash` deduplication.
- **Licence Governance:**
  - `CC-0`, `CC-BY`, `CC-BY-4.0`: Permitted for research, redistribution, and documentation/screenshots with attribution.
  - `CC-BY-NC`, `CC-BY-NC-4.0`, `SCIN-Data-Use-Public-License`: Permitted for local research only; excluded from redistribution, commercial use, and documentation screenshots.
  - Raw images and metadata dumps are git-ignored under `var/cache/derm/`.

---

## 3. Locked-Test Performance & Subgroup Fairness (`n = 118`)

- **Overall (`locked_test`, `n = 118`):**
  - Malignancy AUROC: **`0.8326`** (95% CI `[0.7569, 0.9056]`)
  - Sensitivity @ 90% Specificity: `0.6000` · Sensitivity @ 95% Specificity: `0.5273`
  - Balanced Accuracy: `0.7062` · 9-Class Diagnosis Macro-F1: `0.3023` · Top-3 Accuracy: `0.7034`
  - Calibration: ECE `0.1262`, Brier `0.1753` · Abstention Rate: `35.59%` · Non-Abstained High-Risk Precision: `0.8409`
  - Grad-CAM Lesion-Mask Sanity: `mean L1 shift = 0.6001` (`sanity_passed = true`)
- **Subgroup Disparities & Honest Sample-Size Flags:**
  - **Dataset:** `isic` (`n = 67`) achieves AUROC `0.8841` [`0.7966, 0.9487`] vs. `fitzpatrick17k` (`n = 51`) AUROC `0.7414` [`0.6110, 0.8561`].
  - **Fitzpatrick Skin Tone:** Only `I` (`n = 83`, AUROC `0.8541`) meets `n >= 30`; tones `II` (`8`), `III` (`4`), `IV` (`4`), `V` (`1`), `VI` (`5`), and `UNKNOWN` (`13`) are **flagged `"too few to conclude (< 30)"`**. Dark-skin malignant examples are severely underrepresented across the source corpus (`0.47%` tone V, `0.27%` tone VI, `94.56%` `UNKNOWN`).
  - **Modality:** `clinical` (`n = 91`) achieves AUROC `0.8760` [`0.8082, 0.9349`]; `dermoscopic` (`n = 14`) and `tbp_tile` (`n = 13`) are flagged `"too few to conclude (< 30)"`.

---

## 4. Privacy & Safety Controls

- **EXIF & GPS Stripping:** All uploaded images have EXIF, GPS, XMP, and ICC metadata stripped before storage (`exif_stripped = 1`).
- **Abstention Gate:** Non-skin images (`other`), blurry/underexposed/overexposed/low-resolution images, low-confidence predictions (`< 0.62`), or high out-of-distribution energy (`> -0.15`) automatically abstain (`risk_band = "abstained"`) with bilingual EN/FR explanations.
