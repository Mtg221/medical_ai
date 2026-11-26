# Dermatology Corpus Data Card (`DERM_CORPUS_CARD.md`)

- **Generated (UTC):** `2026-11-16T20:28:56Z`
- **Source Endpoint:** `https://d38rablk9v12ns.cloudfront.net/` (HTTP Basic Auth realm `derm-corpus`)
- **Local Cache Paths:** `var/cache/derm/meta/` (metadata), `var/cache/derm/thumbs/` & `var/cache/derm_thumbs/` (thumbnails) — **git-ignored**
- **Taxonomy Config:** [configs/vision/dx_taxonomy.yaml](../../configs/vision/dx_taxonomy.yaml) (`DERM-DX-HARMONISED-V1`, minimum class count `30`)
- **Metadata Status:** Synced (`573024` records across `573024` unique lesion groups)

> **Clinical & Scientific Safety Notice (EN / FR)**
> - **EN:** Research tool — not a diagnosis. A dermatologist must examine the lesion.
> - **FR :** Outil de recherche — ne constitue pas un diagnostic. Un dermatologue doit examiner la lésion.

---

## 1. Corpus Overview & Lesion / Patient Grouping (`mk`)

| Metric | Value |
|---|---|
| Total Metadata Records (`i`) | `573024` |
| Unique Image IDs (`i`) | `573024` |
| Duplicate Image IDs in Metadata | `0` |
| Unique Lesion / Case Groups (`dataset:mk` or `group_key`) | `573024` |
| Records with Explicit `mk` Lesion/Patient Key | `0` (`0` unique explicit `mk` keys) |
| Quality Flag `ok=True` | `305964` |
| Quality Flag `ok=False` (Rejected) | `267060` |
| Diagnosis vs. `mg` Malignancy Inconsistencies | `0` |

---

## 2. Local Thumbnail Cache & Image Dimension / Perceptual-Hash Audit

Measured directly on locally cached `.webp` thumbnails in `var/cache/derm_thumbs/` (`dataset=isic`):

| Property | Measured Value |
|---|---|
| Cached Thumbnails on Disk | `300` |
| Cached Datasets | `{'isic': 300}` |
| Total Disk Footprint | `6,306,788 bytes` (mean `21,022.6 bytes`/image) |
| Width Range (min / median / max) | `106` / `512.0` / `512` px |
| Height Range (min / median / max) | `99` / `456.5` / `512` px |
| Top Resolutions (`WxH`) | `{'512x512': 83, '512x341': 71, '512x384': 38, '384x512': 34, '512x511': 9, '511x512': 8, '394x394': 2, '257x232': 1, '512x364': 1, '281x281': 1}` |
| 64-bit Perceptual Hash (`dHash`) Evaluated | `300` |
| Unique 64-bit `dHash` Signatures | `299` |
| Exact `dHash` Duplicates in Local Cache | `1` |
| Deterministic Split Distribution (Cached Thumbnails) | `{'train': 211, 'val': 49, 'locked_test': 40}` |

---

## 3. Breakdown by Modality, Source Dataset, Malignancy, Skin Tone, Body Site & Confirmation

### 3.1 By Modality (`clinical`, `dermoscopic`, `tbp_tile`)
| Modality | Count | Share | Flag |
|---|---|---|---|
| `clinical` | `33231` | `5.80%` | adequate (`>= 30`) |
| `dermoscopic` | `121744` | `21.25%` | adequate (`>= 30`) |
| `tbp_tile` | `418049` | `72.95%` | adequate (`>= 30`) |


### 3.2 By Source Dataset (`d`)
| Source Dataset | Count | Share | Flag |
|---|---|---|---|
| `isic` | `549541` | `95.90%` | adequate (`>= 30`) |
| `fitzpatrick17k` | `11394` | `1.99%` | adequate (`>= 30`) |
| `scin` | `10379` | `1.81%` | adequate (`>= 30`) |
| `skindisnet` | `1710` | `0.30%` | adequate (`>= 30`) |


### 3.3 By Malignancy Group (`mg`)
| Malignancy Group | Count | Share | Flag |
|---|---|---|---|
| `unknown` | `39820` | `6.95%` | adequate (`>= 30`) |
| `malignant` | `28740` | `5.02%` | adequate (`>= 30`) |
| `benign` | `501150` | `87.46%` | adequate (`>= 30`) |
| `indeterminate` | `3314` | `0.58%` | adequate (`>= 30`) |


### 3.4 By Fitzpatrick Skin Tone (`st`: `I`–`VI`, `UNKNOWN`)
| Fitzpatrick Skin Tone | Count | Share | Flag |
|---|---|---|---|
| `I` | `4412` | `0.77%` | adequate (`>= 30`) |
| `II` | `11154` | `1.95%` | adequate (`>= 30`) |
| `III` | `6861` | `1.20%` | adequate (`>= 30`) |
| `IV` | `4488` | `0.78%` | adequate (`>= 30`) |
| `V` | `2702` | `0.47%` | adequate (`>= 30`) |
| `VI` | `1532` | `0.27%` | adequate (`>= 30`) |
| `UNKNOWN` | `541875` | `94.56%` | adequate (`>= 30`) |


### 3.5 By Body Site (`bs`)
| Body Site | Count | Share | Flag |
|---|---|---|---|
| `posterior torso` | `139866` | `24.41%` | adequate (`>= 30`) |
| `lower extremity` | `131881` | `23.01%` | adequate (`>= 30`) |
| `anterior torso` | `107446` | `18.75%` | adequate (`>= 30`) |
| `upper extremity` | `92396` | `16.12%` | adequate (`>= 30`) |
| `UNKNOWN` | `60849` | `10.62%` | adequate (`>= 30`) |
| `head/neck` | `27936` | `4.88%` | adequate (`>= 30`) |
| `palms/soles` | `2182` | `0.38%` | adequate (`>= 30`) |
| `leg` | `1581` | `0.28%` | adequate (`>= 30`) |
| `arm` | `1462` | `0.26%` | adequate (`>= 30`) |
| `lateral torso` | `1405` | `0.25%` | adequate (`>= 30`) |
| `other` | `541` | `0.09%` | adequate (`>= 30`) |
| `head or neck` | `447` | `0.08%` | adequate (`>= 30`) |
| `oral/genital` | `372` | `0.06%` | adequate (`>= 30`) |
| `arm, leg` | `304` | `0.05%` | adequate (`>= 30`) |
| `torso front` | `272` | `0.05%` | adequate (`>= 30`) |
| `back of hand` | `253` | `0.04%` | adequate (`>= 30`) |
| `foot top or side` | `220` | `0.04%` | adequate (`>= 30`) |
| `torso back` | `195` | `0.03%` | adequate (`>= 30`) |
| `leg, foot top or side` | `111` | `0.02%` | adequate (`>= 30`) |
| `torso front, torso back` | `107` | `0.02%` | adequate (`>= 30`) |
| `palm` | `94` | `0.02%` | adequate (`>= 30`) |
| `genitalia or groin` | `86` | `0.02%` | adequate (`>= 30`) |
| `arm, back of hand` | `86` | `0.02%` | adequate (`>= 30`) |
| `head or neck, arm` | `83` | `0.01%` | adequate (`>= 30`) |
| `buttocks` | `77` | `0.01%` | adequate (`>= 30`) |
| `arm, torso front, leg` | `63` | `0.01%` | adequate (`>= 30`) |
| `arm, other` | `63` | `0.01%` | adequate (`>= 30`) |
| `arm, torso front` | `59` | `0.01%` | adequate (`>= 30`) |
| `arm, torso front, torso back, leg` | `54` | `0.01%` | adequate (`>= 30`) |
| `arm, back of hand, leg` | `51` | `0.01%` | adequate (`>= 30`) |
| `head or neck, arm, leg` | `50` | `0.01%` | adequate (`>= 30`) |
| `arm, torso front, torso back, buttocks, leg` | `48` | `0.01%` | adequate (`>= 30`) |
| `head or neck, other` | `47` | `0.01%` | adequate (`>= 30`) |
| `palm, back of hand` | `45` | `0.01%` | adequate (`>= 30`) |
| `buttocks, leg` | `44` | `0.01%` | adequate (`>= 30`) |
| `leg, other` | `43` | `0.01%` | adequate (`>= 30`) |
| `head or neck, torso front` | `42` | `0.01%` | adequate (`>= 30`) |
| `head or neck, torso front, torso back` | `40` | `0.01%` | adequate (`>= 30`) |
| `head or neck, arm, torso front, torso back, leg` | `38` | `0.01%` | adequate (`>= 30`) |
| `head or neck, arm, torso front, torso back` | `37` | `0.01%` | adequate (`>= 30`) |
| `torso front, leg` | `35` | `0.01%` | adequate (`>= 30`) |
| `head or neck, arm, palm, back of hand, torso front, torso back, genitalia or groin, buttocks, leg, foot top or side, foot sole` | `34` | `0.01%` | adequate (`>= 30`) |
| `arm, torso back, leg` | `33` | `0.01%` | adequate (`>= 30`) |
| `head or neck, arm, torso front, torso back, buttocks, leg` | `33` | `0.01%` | adequate (`>= 30`) |
| `foot sole` | `32` | `0.01%` | adequate (`>= 30`) |
| `head or neck, arm, torso front` | `31` | `0.01%` | adequate (`>= 30`) |
| `arm, torso front, torso back` | `29` | `0.01%` | too few to conclude (`< 30`) |
| `arm, buttocks, leg` | `27` | `0.00%` | too few to conclude (`< 30`) |
| `arm, leg, other` | `27` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, back of hand, leg` | `25` | `0.00%` | too few to conclude (`< 30`) |
| `foot top or side, foot sole` | `23` | `0.00%` | too few to conclude (`< 30`) |
| `torso front, torso back, buttocks` | `23` | `0.00%` | too few to conclude (`< 30`) |
| `arm, torso front, torso back, buttocks, leg, foot top or side` | `23` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, torso front, torso back` | `22` | `0.00%` | too few to conclude (`< 30`) |
| `arm, palm` | `22` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, leg, other` | `21` | `0.00%` | too few to conclude (`< 30`) |
| `torso front, other` | `20` | `0.00%` | too few to conclude (`< 30`) |
| `back of hand, other` | `19` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, torso front, torso back, buttocks, leg, foot top or side` | `19` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, other` | `19` | `0.00%` | too few to conclude (`< 30`) |
| `torso front, torso back, leg` | `18` | `0.00%` | too few to conclude (`< 30`) |
| `arm, palm, back of hand` | `18` | `0.00%` | too few to conclude (`< 30`) |
| `arm, torso back` | `18` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, back of hand` | `18` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, torso front, leg` | `16` | `0.00%` | too few to conclude (`< 30`) |
| `arm, torso front, torso back, genitalia or groin, buttocks, leg` | `16` | `0.00%` | too few to conclude (`< 30`) |
| `genitalia or groin, leg` | `16` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, torso front, torso back, leg` | `16` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, leg` | `16` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, buttocks, leg` | `15` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, back of hand, torso front, torso back, buttocks, leg, foot top or side` | `15` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, leg, foot top or side` | `14` | `0.00%` | too few to conclude (`< 30`) |
| `torso back, leg` | `14` | `0.00%` | too few to conclude (`< 30`) |
| `arm, leg, foot top or side` | `14` | `0.00%` | too few to conclude (`< 30`) |
| `arm, torso back, buttocks, leg` | `14` | `0.00%` | too few to conclude (`< 30`) |
| `buttocks, leg, other` | `14` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, back of hand, torso front, torso back` | `13` | `0.00%` | too few to conclude (`< 30`) |
| `leg, foot top or side, other` | `13` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, torso back, leg` | `12` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, torso front, leg` | `12` | `0.00%` | too few to conclude (`< 30`) |
| `torso front, torso back, other` | `12` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, back of hand, torso front` | `12` | `0.00%` | too few to conclude (`< 30`) |
| `back of hand, leg, foot top or side` | `12` | `0.00%` | too few to conclude (`< 30`) |
| `arm, genitalia or groin, leg` | `12` | `0.00%` | too few to conclude (`< 30`) |
| `buttocks, leg, foot top or side` | `11` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, torso front, other` | `11` | `0.00%` | too few to conclude (`< 30`) |
| `genitalia or groin, buttocks` | `11` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, torso back` | `11` | `0.00%` | too few to conclude (`< 30`) |
| `arm, torso front, other` | `11` | `0.00%` | too few to conclude (`< 30`) |
| `torso back, other` | `10` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, palm, back of hand, torso front, torso back, genitalia or groin, buttocks, leg, foot top or side` | `10` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, back of hand, torso front, torso back, genitalia or groin, buttocks, leg, foot top or side` | `10` | `0.00%` | too few to conclude (`< 30`) |
| `torso back, buttocks` | `10` | `0.00%` | too few to conclude (`< 30`) |
| `palm, other` | `10` | `0.00%` | too few to conclude (`< 30`) |
| `arm, torso back, buttocks, leg, foot top or side` | `9` | `0.00%` | too few to conclude (`< 30`) |
| `arm, torso front, leg, other` | `9` | `0.00%` | too few to conclude (`< 30`) |
| `torso back, buttocks, leg` | `9` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, torso front, leg, foot top or side` | `9` | `0.00%` | too few to conclude (`< 30`) |
| `torso front, torso back, buttocks, leg` | `9` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, leg, other` | `9` | `0.00%` | too few to conclude (`< 30`) |
| `arm, foot top or side` | `9` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, palm, back of hand, torso front, torso back, leg` | `9` | `0.00%` | too few to conclude (`< 30`) |
| `foot top or side, other` | `9` | `0.00%` | too few to conclude (`< 30`) |
| `torso front, genitalia or groin` | `8` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, palm, back of hand` | `8` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, back of hand, torso front, torso back, genitalia or groin, buttocks, leg, foot top or side, foot sole` | `8` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, back of hand, torso front, torso back, leg` | `8` | `0.00%` | too few to conclude (`< 30`) |
| `torso front, leg, foot top or side` | `8` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, back of hand, torso front, torso back, leg, foot top or side` | `8` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, torso front, torso back, genitalia or groin, leg, foot top or side` | `8` | `0.00%` | too few to conclude (`< 30`) |
| `arm, torso back, leg, foot top or side` | `8` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, torso back, leg` | `8` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, palm` | `7` | `0.00%` | too few to conclude (`< 30`) |
| `arm, palm, back of hand, leg, foot top or side, foot sole` | `7` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, palm, torso front` | `7` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, buttocks, leg` | `7` | `0.00%` | too few to conclude (`< 30`) |
| `back of hand, foot top or side` | `7` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, torso front, torso back, buttocks, leg, other` | `7` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, torso back` | `7` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, torso front, genitalia or groin` | `7` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, torso front, torso back, leg` | `7` | `0.00%` | too few to conclude (`< 30`) |
| `arm, torso front, leg, foot top or side` | `7` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, back of hand, torso back, genitalia or groin, buttocks, leg` | `7` | `0.00%` | too few to conclude (`< 30`) |
| `arm, torso back, other` | `7` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, torso front, torso back, genitalia or groin, buttocks, leg` | `7` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, palm, back of hand, leg` | `6` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, back of hand, torso front, leg` | `6` | `0.00%` | too few to conclude (`< 30`) |
| `torso front, torso back, buttocks, leg, foot top or side` | `6` | `0.00%` | too few to conclude (`< 30`) |
| `torso front, torso back, leg, other` | `6` | `0.00%` | too few to conclude (`< 30`) |
| `leg, foot sole` | `6` | `0.00%` | too few to conclude (`< 30`) |
| `genitalia or groin, leg, other` | `6` | `0.00%` | too few to conclude (`< 30`) |
| `torso front, torso back, genitalia or groin` | `6` | `0.00%` | too few to conclude (`< 30`) |
| `back of hand, leg` | `6` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, torso front, leg, foot top or side` | `6` | `0.00%` | too few to conclude (`< 30`) |
| `arm, torso front, torso back, leg, foot top or side` | `6` | `0.00%` | too few to conclude (`< 30`) |
| `genitalia or groin, other` | `6` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, back of hand, torso front, torso back, buttocks` | `6` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, other` | `6` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, torso front, torso back, genitalia or groin, buttocks, leg, foot top or side` | `6` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, genitalia or groin, leg, foot top or side` | `6` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, palm, back of hand, torso front, torso back, buttocks, leg, foot top or side` | `6` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, torso front, other` | `6` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, leg, foot top or side` | `6` | `0.00%` | too few to conclude (`< 30`) |
| `arm, palm, leg` | `6` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, palm, back of hand, torso front, leg` | `6` | `0.00%` | too few to conclude (`< 30`) |
| `arm, torso front, buttocks` | `6` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, palm, back of hand, torso front, torso back, genitalia or groin, buttocks, leg` | `6` | `0.00%` | too few to conclude (`< 30`) |
| `arm, torso front, torso back, genitalia or groin, leg` | `6` | `0.00%` | too few to conclude (`< 30`) |
| `arm, torso front, torso back, buttocks` | `6` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, leg, foot top or side, other` | `6` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, palm, torso front, torso back, leg, other` | `6` | `0.00%` | too few to conclude (`< 30`) |
| `palm, back of hand, foot top or side, foot sole` | `6` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, torso front, torso back, genitalia or groin, buttocks, leg` | `6` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, back of hand, torso front, torso back, buttocks, leg, foot top or side, other` | `5` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, back of hand, torso front, torso back, leg, foot top or side` | `5` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, palm, back of hand, leg, foot sole` | `5` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, palm, leg` | `5` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, back of hand, leg, foot top or side` | `5` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, torso front, leg` | `5` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, torso front, torso back, genitalia or groin` | `5` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, buttocks, leg, foot top or side` | `5` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, foot top or side, other` | `5` | `0.00%` | too few to conclude (`< 30`) |
| `arm, buttocks` | `5` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, torso front, buttocks, leg, foot top or side` | `5` | `0.00%` | too few to conclude (`< 30`) |
| `arm, leg, foot top or side, other` | `5` | `0.00%` | too few to conclude (`< 30`) |
| `arm, palm, back of hand, foot top or side, foot sole` | `5` | `0.00%` | too few to conclude (`< 30`) |
| `palm, foot sole` | `5` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, torso front, genitalia or groin, leg` | `4` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, genitalia or groin` | `4` | `0.00%` | too few to conclude (`< 30`) |
| `torso front, leg, other` | `4` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, torso back, buttocks, leg` | `4` | `0.00%` | too few to conclude (`< 30`) |
| `arm, palm, back of hand, foot top or side` | `4` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, back of hand` | `4` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, palm, back of hand, torso front, torso back` | `4` | `0.00%` | too few to conclude (`< 30`) |
| `torso front, buttocks` | `4` | `0.00%` | too few to conclude (`< 30`) |
| `arm, palm, torso front` | `4` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, torso front, buttocks, leg` | `4` | `0.00%` | too few to conclude (`< 30`) |
| `arm, palm, back of hand, leg` | `4` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, back of hand, torso front, torso back, genitalia or groin, buttocks, leg` | `4` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, torso front` | `4` | `0.00%` | too few to conclude (`< 30`) |
| `back of hand, genitalia or groin` | `4` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, torso front, torso back, buttocks` | `4` | `0.00%` | too few to conclude (`< 30`) |
| `palm, foot top or side, foot sole` | `4` | `0.00%` | too few to conclude (`< 30`) |
| `arm, palm, foot top or side` | `4` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, torso front, torso back, buttocks, leg` | `4` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, torso front, torso back, other` | `4` | `0.00%` | too few to conclude (`< 30`) |
| `torso back, leg, other` | `4` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, buttocks, leg` | `4` | `0.00%` | too few to conclude (`< 30`) |
| `genitalia or groin, buttocks, leg` | `4` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, torso front, torso back, buttocks, leg, foot top or side` | `4` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, torso front, buttocks, leg` | `4` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, torso front, torso back, buttocks, leg, foot top or side, other` | `4` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, torso front, buttocks, other` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, palm, back of hand, torso front, genitalia or groin, leg` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, torso front, genitalia or groin, foot top or side` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `torso front, torso back, genitalia or groin, buttocks, leg` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, torso front, torso back, buttocks, leg, other` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, torso back` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, torso front, torso back, genitalia or groin, buttocks, leg, foot top or side, foot sole` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, palm, torso front, genitalia or groin, buttocks, leg, foot sole` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, palm, back of hand, torso front, torso back, buttocks, leg, foot top or side, foot sole` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, torso front, torso back, leg, foot top or side` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, leg, foot sole` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, palm, buttocks, leg, foot top or side, foot sole, other` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, palm, back of hand, torso front, torso back, genitalia or groin, leg, other` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, torso front, buttocks, leg` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, leg, foot top or side, other` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, foot sole` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, palm, back of hand, torso front, torso back, buttocks, leg, foot top or side, foot sole, other` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `torso front, genitalia or groin, leg` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, torso front, torso back, leg, foot top or side` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, torso front, torso back, leg, other` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, torso front, torso back, genitalia or groin` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, palm, back of hand, torso front, buttocks, leg, foot top or side, foot sole` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, back of hand, torso front, leg` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, genitalia or groin, buttocks, leg` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, back of hand, torso front, genitalia or groin, leg` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, palm, back of hand, torso front, genitalia or groin, leg` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `torso front, torso back, genitalia or groin, buttocks` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, palm, back of hand, torso front, torso back, buttocks, leg, foot top or side, foot sole` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, torso front, torso back, leg, foot top or side, foot sole, other` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, palm, back of hand, torso back, buttocks, leg, foot top or side` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, back of hand, leg, foot top or side` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, torso back, genitalia or groin, buttocks, leg, foot top or side` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, genitalia or groin` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, palm, back of hand, torso front, torso back, leg, foot top or side, foot sole` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, buttocks, foot top or side` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, buttocks, leg, other` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, torso back, buttocks, leg, foot top or side, foot sole` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `palm, leg, foot top or side, foot sole` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, palm, back of hand, torso back, genitalia or groin` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, palm, leg, foot sole` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, back of hand, torso front, torso back, genitalia or groin, buttocks, leg, foot top or side, foot sole, other` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, torso back, genitalia or groin, leg` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, back of hand, torso front, leg, foot top or side` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, palm, torso front, torso back, buttocks` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, palm, torso front, torso back, buttocks, leg, foot top or side` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, torso front, torso back, leg, other` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, back of hand, torso front, torso back, buttocks, leg` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, back of hand, torso front, buttocks, leg` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, palm, back of hand, torso front, genitalia or groin, leg, foot sole` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, palm, back of hand, leg` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, back of hand, torso front, genitalia or groin, leg, foot top or side` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, leg, foot top or side, foot sole, other` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, torso front, torso back, buttocks, leg, foot top or side, other` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, palm, back of hand, torso front, torso back, genitalia or groin, buttocks, leg, foot top or side, other` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, back of hand, torso front, torso back, genitalia or groin, leg, foot top or side, foot sole, other` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, palm, back of hand, torso front, torso back, genitalia or groin` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `back of hand, torso front` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, torso front, torso back, buttocks, leg` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, back of hand, torso front, torso back, genitalia or groin, buttocks, leg, foot top or side, other` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, torso front, genitalia or groin, leg, foot top or side` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, genitalia or groin, leg, foot sole` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, genitalia or groin, buttocks, leg` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, back of hand, buttocks, leg` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, palm, back of hand, torso front, genitalia or groin, buttocks, leg, foot top or side, other` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `buttocks, other` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, back of hand, buttocks` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, back of hand, torso back, leg` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, leg, foot top or side, foot sole` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, leg, foot top or side` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, torso front, torso back, genitalia or groin, buttocks, leg, other` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, back of hand, torso front, torso back, genitalia or groin, leg, foot top or side` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, torso front, torso back, genitalia or groin, leg, other` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, torso front, torso back, genitalia or groin, buttocks, leg` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, torso front, genitalia or groin, leg` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, genitalia or groin, leg, foot top or side` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, back of hand, torso front, torso back, genitalia or groin, buttocks, leg, foot top or side` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `torso front, buttocks, leg` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, torso front, leg, other` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, torso front, leg, foot top or side, other` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, torso back, leg, foot top or side` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, palm, back of hand, genitalia or groin, foot top or side` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, buttocks, other` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, palm, leg, foot sole` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `palm, foot top or side` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, palm, back of hand, torso front, torso back, genitalia or groin, leg` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, palm, back of hand, torso front, leg, foot top or side, foot sole` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, back of hand, torso back, leg, foot top or side` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, palm, back of hand, buttocks, leg, foot top or side` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, torso front, buttocks, other` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, buttocks` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, palm, torso back, foot top or side, foot sole` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, torso front, torso back, genitalia or groin, buttocks, leg, foot top or side` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, palm, back of hand, genitalia or groin, leg, other` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `palm, back of hand, genitalia or groin` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, palm, torso front, genitalia or groin, buttocks, leg` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, torso back, buttocks` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, palm, torso front, other` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, genitalia or groin` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, torso front, torso back, genitalia or groin, leg, foot top or side` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, palm, torso front, torso back, genitalia or groin, buttocks, foot top or side, foot sole` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, torso front, buttocks, leg` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, torso back, other` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `torso front, leg, foot sole` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, torso front, torso back, genitalia or groin, leg, foot top or side, other` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, buttocks, leg, other` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, palm, torso front, leg, foot top or side` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `foot sole, other` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, genitalia or groin, leg, foot top or side, foot sole` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, palm, back of hand, torso front, torso back, genitalia or groin, leg, foot top or side, foot sole` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, palm, back of hand, torso front, torso back` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, torso back, genitalia or groin, leg, foot sole` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, back of hand, torso front, torso back, genitalia or groin, buttocks, leg, other` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, palm, back of hand, foot top or side, other` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, torso back, foot top or side` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, back of hand, torso back, other` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, genitalia or groin, buttocks, leg, foot top or side` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, torso back, buttocks, leg` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, palm, back of hand, genitalia or groin, buttocks, foot sole` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, back of hand, torso back, leg, other` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, torso back, leg, other` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, palm, back of hand, torso back, buttocks, leg` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `arm, palm, back of hand, leg, foot top or side` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `palm, leg, foot top or side` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `back of hand, torso back` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, palm` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, palm, torso front, genitalia or groin, leg` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, palm, back of hand, torso front, torso back, leg, foot top or side, foot sole, other` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, foot sole` | `3` | `0.00%` | too few to conclude (`< 30`) |
| `back of hand, torso front, buttocks` | `2` | `0.00%` | too few to conclude (`< 30`) |
| `back of hand, genitalia or groin, leg, other` | `2` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, genitalia or groin, buttocks, leg` | `2` | `0.00%` | too few to conclude (`< 30`) |
| `palm, genitalia or groin, foot top or side` | `2` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, buttocks, leg, foot top or side` | `2` | `0.00%` | too few to conclude (`< 30`) |
| `arm, palm, torso front, torso back, leg, foot top or side` | `2` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, torso front, torso back, buttocks, other` | `2` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, torso back, leg, foot top or side, foot sole` | `2` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, torso back, leg` | `2` | `0.00%` | too few to conclude (`< 30`) |
| `back of hand, leg, foot top or side, foot sole` | `2` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, torso front, genitalia or groin` | `2` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, back of hand, leg` | `2` | `0.00%` | too few to conclude (`< 30`) |
| `palm, back of hand, foot top or side` | `2` | `0.00%` | too few to conclude (`< 30`) |
| `palm, back of hand, other` | `2` | `0.00%` | too few to conclude (`< 30`) |
| `arm, palm, back of hand, buttocks, leg, foot top or side, foot sole` | `2` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, back of hand, torso front, torso back, other` | `2` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, torso back, buttocks` | `2` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, back of hand, torso front, genitalia or groin, buttocks, leg, foot top or side, foot sole` | `2` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, buttocks, leg, foot sole, other` | `2` | `0.00%` | too few to conclude (`< 30`) |
| `back of hand, foot top or side, other` | `2` | `0.00%` | too few to conclude (`< 30`) |
| `arm, torso front, genitalia or groin, leg` | `2` | `0.00%` | too few to conclude (`< 30`) |
| `leg, foot top or side, foot sole` | `2` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, back of hand, torso back, genitalia or groin, buttocks, leg, other` | `2` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, palm, back of hand, torso front, torso back, buttocks, leg` | `2` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, torso front, torso back, buttocks` | `2` | `0.00%` | too few to conclude (`< 30`) |
| `back of hand, foot top or side, foot sole` | `2` | `0.00%` | too few to conclude (`< 30`) |
| `arm, genitalia or groin, buttocks, other` | `2` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, back of hand, torso front, torso back, foot top or side` | `2` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, buttocks` | `2` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, back of hand, torso back` | `2` | `0.00%` | too few to conclude (`< 30`) |
| `torso front, torso back, leg, foot top or side` | `2` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, torso front, torso back, genitalia or groin, buttocks` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, torso front, torso back, leg, foot top or side, foot sole` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, torso front, torso back, genitalia or groin, other` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `palm, genitalia or groin, buttocks, leg` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `arm, palm, buttocks, leg, foot top or side` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `arm, genitalia or groin` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `arm, buttocks, leg, foot top or side` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, torso front, torso back, genitalia or groin, buttocks, leg, other` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `back of hand, foot sole` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, torso front, buttocks` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `arm, palm, torso front, torso back, buttocks, leg, foot top or side` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, buttocks, leg, other` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, back of hand, other` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, back of hand, other` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, foot top or side` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `palm, back of hand, leg, foot top or side` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `genitalia or groin, foot top or side, other` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `arm, torso back, genitalia or groin, buttocks, leg` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, torso front, torso back, genitalia or groin, leg` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `arm, genitalia or groin, buttocks` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, torso front, genitalia or groin, buttocks, leg` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `arm, genitalia or groin, other` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, torso front, torso back, buttocks` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `arm, torso front, torso back, genitalia or groin, buttocks, leg, foot top or side` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, torso front, torso back, leg, foot top or side` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, palm, genitalia or groin` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `torso back, genitalia or groin` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `arm, torso back, buttocks, leg, other` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, torso back, buttocks, leg` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `arm, foot top or side, other` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, palm, back of hand, torso front, torso back, genitalia or groin, buttocks, leg, foot top or side, foot sole, other` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, leg, other` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `arm, leg, foot top or side, foot sole` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, torso front, foot top or side` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, palm, back of hand, buttocks, leg, other` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, genitalia or groin, other` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, palm, back of hand, torso front, torso back, foot top or side` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `palm, back of hand, leg` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, torso front, torso back, leg, foot top or side, foot sole` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `back of hand, torso front, leg` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, genitalia or groin, buttocks` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `arm, palm, torso back, leg` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `arm, back of hand, buttocks, other` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `back of hand, leg, foot sole` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, palm, back of hand, genitalia or groin, leg` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, genitalia or groin, buttocks` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `head or neck, arm, torso back, foot top or side` | `1` | `0.00%` | too few to conclude (`< 30`) |
| `palm, back of hand, leg, foot top or side, other` | `1` | `0.00%` | too few to conclude (`< 30`) |


### 3.6 By Confirmation Method (`cf`)
| Confirmation Method | Count | Share | Flag |
|---|---|---|---|
| `UNKNOWN` | `407884` | `71.18%` | adequate (`>= 30`) |
| `histopathology` | `49895` | `8.71%` | adequate (`>= 30`) |
| `single contributor clinical assessment` | `33928` | `5.92%` | adequate (`>= 30`) |
| `serial imaging showing no change` | `31536` | `5.50%` | adequate (`>= 30`) |
| `single image expert consensus` | `26631` | `4.65%` | adequate (`>= 30`) |
| `atlas image (not biopsy-confirmed)` | `11394` | `1.99%` | adequate (`>= 30`) |
| `dermatologist consensus` | `10379` | `1.81%` | adequate (`>= 30`) |
| `confocal microscopy with consensus dermoscopy` | `1377` | `0.24%` | adequate (`>= 30`) |


---

## 4. Harmonised Diagnosis Taxonomy (`configs/vision/dx_taxonomy.yaml`)

Classes with fewer than `30` records across the full corpus are automatically collapsed into `other`. Every class specifies its canonical `malignancy_group` (`benign` or `malignant`) and is checked against the record's raw `mg` attribute.

| Harmonised Class | Malignancy Group | ICD-10 | Label (EN / FR) | Metadata Count | Representative Raw `dx` Aliases |
|---|---|---|---|---|---|
| `melanoma` | `malignant` | `C43.9` | Melanoma / Mélanome | `432` | `melanoma`, `mel`, `malignant melanoma`, `superficial spreading melanoma`, `superficial spreading melanoma ssm`, `nodular melanoma` |
| `basal_cell_carcinoma` | `malignant` | `C44.91` | Basal Cell Carcinoma (BCC) / Carcinome basocellulaire (CBC) | `512` | `basal cell carcinoma`, `basal cell carcinoma morpheiform`, `solid cystic basal cell carcinoma`, `bcc`, `carcinoma basal cell`, `nodular basal cell carcinoma` |
| `squamous_cell_carcinoma` | `malignant` | `C44.92` | Squamous Cell Carcinoma (SCC) / Carcinome épidermoïde (CE) | `388` | `squamous cell carcinoma`, `scc`, `squamous cell carcinoma in situ`, `sccis`, `bowen's disease`, `bowen disease` |
| `cutaneous_lymphoma_sarcoma` | `malignant` | `C84.0` | Cutaneous Lymphoma / Kaposi Sarcoma / Lymphome cutané / Sarcome de Kaposi | `256` | `mycosis fungoides`, `kaposi sarcoma`, `cutaneous t-cell lymphoma`, `cutaneous lymphoma` |
| `actinic_keratosis` | `benign` | `L57.0` | Actinic Keratosis / Porokeratosis / Kératose actinique / Porokératose | `321` | `actinic keratosis`, `ak`, `akiec`, `solar keratosis`, `actinic cheilitis`, `senile keratosis` |
| `nevus` | `benign` | `D22.9` | Melanocytic & Epidermal Nevus / Nævus mélanocytaire et épidermique | `440` | `nevus`, `naevus`, `nevi`, `nv`, `melanocytic nevus`, `melanocytic nevi` |
| `seborrheic_keratosis` | `benign` | `L82.1` | Seborrheic Keratosis / Solar Lentigo / Kératose séborrhéique / Lentigo solaire | `53` | `seborrheic keratosis`, `seborrhoeic keratosis`, `sek`, `bkl`, `benign keratosis`, `benign keratosis-like lesions` |
| `dermatofibroma` | `benign` | `D23.9` | Dermatofibroma / Dermatofibrome | `71` | `dermatofibroma`, `df`, `fibrous histiocytoma`, `benign fibrous histiocytoma`, `histiocytoma` |
| `vascular_lesion` | `benign` | `D18.0` | Vascular Lesion / Lésion vasculaire | `445` | `vascular lesion`, `vascular lesions`, `vasc`, `hemangioma`, `haemangioma`, `angioma` |
| `benign_cyst_adnexal` | `benign` | `L72.9` | Benign Cyst / Adnexal Lesion / Kyste bénin / Lésion annexielle | `542` | `milia`, `pilar cyst`, `epidermoid cyst`, `mucous cyst`, `syringoma`, `pilomatricoma` |
| `inflammatory_infectious_dermatosis` | `benign` | `L30.9` | Inflammatory / Infectious Dermatosis / Dermatose inflammatoire ou infectieuse | `11667` | `psoriasis`, `eczema`, `ec`, `atopic dermatitis`, `ad`, `allergic contact dermatitis` |
| `benign_unspecified` | `benign` | `D23.9` | Benign Cutaneous Lesion (Unspecified Subtype) / Lésion cutanée bénigne (sous-type non spécifié) | `499643` | `benign` |
| `malignant_unspecified` | `malignant` | `C44.90` | Malignant Cutaneous Lesion (Unspecified Subtype) / Lésion cutanée maligne (sous-type non spécifié) | `27057` | `malignant` |
| `other` | `any` | `L98.9` | Other / Indeterminate / Unspecified / Autre / Indéterminé / Non spécifié | `31197` | `other`, `unknown`, `indeterminate`, `none`, `{}` |


### Top Raw Diagnosis Strings (`dx`)
| Raw Diagnosis (`dx`) | Count | Share | Flag |
|---|---|---|---|
| `Benign` | `499643` | `87.19%` | adequate (`>= 30`) |
| `Malignant` | `27057` | `4.72%` | adequate (`>= 30`) |
| `UNKNOWN` | `19527` | `3.41%` | adequate (`>= 30`) |
| `{}` | `3876` | `0.68%` | adequate (`>= 30`) |
| `Indeterminate` | `3314` | `0.58%` | adequate (`>= 30`) |
| `psoriasis` | `534` | `0.09%` | adequate (`>= 30`) |
| `CD` | `477` | `0.08%` | adequate (`>= 30`) |
| `EC` | `466` | `0.08%` | adequate (`>= 30`) |
| `basal cell carcinoma` | `412` | `0.07%` | adequate (`>= 30`) |
| `lichen planus` | `377` | `0.07%` | adequate (`>= 30`) |
| `squamous cell carcinoma` | `372` | `0.06%` | adequate (`>= 30`) |
| `SC` | `343` | `0.06%` | adequate (`>= 30`) |
| `allergic contact dermatitis` | `314` | `0.05%` | adequate (`>= 30`) |
| `acne vulgaris` | `299` | `0.05%` | adequate (`>= 30`) |
| `{'Eczema': 1.0}` | `282` | `0.05%` | adequate (`>= 30`) |
| `TC` | `275` | `0.05%` | adequate (`>= 30`) |
| `lupus erythematosus` | `257` | `0.04%` | adequate (`>= 30`) |
| `sarcoidosis` | `257` | `0.04%` | adequate (`>= 30`) |
| `scleroderma` | `246` | `0.04%` | adequate (`>= 30`) |
| `neutrophilic dermatoses` | `244` | `0.04%` | adequate (`>= 30`) |
| `photodermatoses` | `242` | `0.04%` | adequate (`>= 30`) |
| `scabies` | `203` | `0.04%` | adequate (`>= 30`) |
| `folliculitis` | `182` | `0.03%` | adequate (`>= 30`) |
| `eczema` | `176` | `0.03%` | adequate (`>= 30`) |
| `melanoma` | `176` | `0.03%` | adequate (`>= 30`) |
| `pityriasis rubra pilaris` | `176` | `0.03%` | adequate (`>= 30`) |
| `{'Urticaria': 1.0}` | `160` | `0.03%` | adequate (`>= 30`) |
| `drug eruption` | `157` | `0.03%` | adequate (`>= 30`) |
| `erythema multiforme` | `154` | `0.03%` | adequate (`>= 30`) |
| `mycosis fungoides` | `143` | `0.02%` | adequate (`>= 30`) |
| `actinic keratosis` | `142` | `0.02%` | adequate (`>= 30`) |
| `dermatomyositis` | `135` | `0.02%` | adequate (`>= 30`) |
| `urticaria` | `134` | `0.02%` | adequate (`>= 30`) |
| `dariers disease` | `130` | `0.02%` | adequate (`>= 30`) |
| `granuloma annulare` | `130` | `0.02%` | adequate (`>= 30`) |
| `pityriasis rosea` | `123` | `0.02%` | adequate (`>= 30`) |
| `acne` | `121` | `0.02%` | adequate (`>= 30`) |
| `nematode infection` | `113` | `0.02%` | adequate (`>= 30`) |
| `seborrheic dermatitis` | `113` | `0.02%` | adequate (`>= 30`) |
| `vitiligo` | `113` | `0.02%` | adequate (`>= 30`) |
| `neurofibromatosis` | `109` | `0.02%` | adequate (`>= 30`) |
| `juvenile xanthogranuloma` | `104` | `0.02%` | adequate (`>= 30`) |
| `superficial spreading melanoma ssm` | `104` | `0.02%` | adequate (`>= 30`) |
| `telangiectases` | `100` | `0.02%` | adequate (`>= 30`) |
| `kaposi sarcoma` | `98` | `0.02%` | adequate (`>= 30`) |
| `pyogenic granuloma` | `95` | `0.02%` | adequate (`>= 30`) |
| `porokeratosis actinic` | `93` | `0.02%` | adequate (`>= 30`) |
| `malignant melanoma` | `90` | `0.02%` | adequate (`>= 30`) |
| `urticaria pigmentosa` | `90` | `0.02%` | adequate (`>= 30`) |
| `stevens johnson syndrome` | `89` | `0.02%` | adequate (`>= 30`) |


---

## 5. Licence Governance & Privacy Rules (`lc`)

| Licence Category | Rule / Permission | Metadata Count |
|---|---|---|
| **Research Allowed** (`CC0`, `CC-BY*`, `CC-BY-NC*`, `SYNTHETIC-INTERNAL`) | Permitted for local model training & evaluation | `573024` |
| **Redistribution Allowed** (`CC0`, `CC-BY`, `CC-BY-SA`) | Permitted for redistribution with attribution | `305964` |
| **Commercial Use Allowed** (`CC0`, `CC-BY`, `CC-BY-SA`) | Permitted for commercial use | `305964` |
| **Safe for Docs & Screenshots** (`CC0`, `CC-BY`, `SYNTHETIC-INTERNAL`) | Only these (with attribution) or synthetic images may appear in docs/screenshots | `305964` |
| **Excluded / Unknown Licence** | Excluded from training & evaluation (`require_research_licence=True`) | `0` |

### Counts per Raw Licence (`lc`)
| Raw Licence (`lc`) | Count | Share | Flag |
|---|---|---|---|
| `CC-BY` | `255503` | `44.59%` | adequate (`>= 30`) |
| `CC-BY-NC` | `245287` | `42.81%` | adequate (`>= 30`) |
| `CC-0` | `48751` | `8.51%` | adequate (`>= 30`) |
| `CC-BY-NC-4.0` | `11394` | `1.99%` | adequate (`>= 30`) |
| `SCIN-Data-Use-Public-License` | `10379` | `1.81%` | adequate (`>= 30`) |
| `CC-BY-4.0` | `1710` | `0.30%` | adequate (`>= 30`) |


**Privacy & Storage Enforcement:**
- Raw images, thumbnails, and metadata dumps are stored strictly inside `var/cache/derm/` (git-ignored) and are **never committed to git**.
- Clinical photographs may contain identifiable facial features or tattoos: full-size images are restricted to authenticated sessions, and list views render thumbnails only.

---

## 6. Leak-Free Lesion Split & Perceptual-Hash Deduplication Audit

- **Split Rule:** Deterministic SHA-256 hash of `group_key = f"{dataset}:{mk}"` (`70%` `train`, `15%` `val`, `15%` `locked_test`), ensuring all photographs of the same lesion/patient stay in the exact same split.
- **Cross-Split Near-Duplicate Gate:** 64-bit horizontal difference perceptual hash (`dHash`) computed on thumbnails; any secondary image sharing a `dHash` across splits is excluded before evaluation.
- **Audit Status:** **`PASSED (0 overlap)`**
  - `train ∩ val` lesion overlap: `0`
  - `train ∩ locked_test` lesion overlap: `0`
  - `val ∩ locked_test` lesion overlap: `0`
  - Cross-split `dHash` duplicates excluded: `0`
