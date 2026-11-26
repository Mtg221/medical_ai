# Dermatology Corpus Data Card (v3 — Lesion-Similarity Grouped & Tone×Malignancy Stratified)

- **Frozen test list:** `configs/vision/locked_test_v3.txt` (`n=1624`, `sha256=8191dd603b851ff551934666d88736a70a6bff1d7194de91554bddfb4a768209`)
- **Val list:** `configs/vision/val_v3.txt` (`n=1497`)
- **Cross-split lesion group overlap:** `0` (`configs/vision/lesion_groups_v3.json`)
- **Superseded notice:** `locked_test_v2` is marked superseded because images of the same lesion could cross splits (W1); v2 numbers are retained for historical comparison only.

## 1. Split Summary (Canonical `compute_split_counts`)

| Split | Total | Benign | Malignant | Masked (`mg`) | Clinical | Dermoscopic | TBP Tile |
|---|---|---|---|---|---|---|---|
| `train` | 31565 | 20630 | 8125 | 2810 | 8003 | 22107 | 1455 |
| `val` | 1497 | 900 | 597 | 0 | 352 | 850 | 295 |
| `locked_test_v3` | 1624 | 877 | 747 | 0 | 574 | 700 | 350 |

## 2. `locked_test_v3` Tone × Malignancy & Modality × Malignancy

| Tone | Benign (`locked_test_v3`) | Malignant (`locked_test_v3`) | Note |
|---|---|---|---|
| `I` | 75 | 75 | adequate (>=75 each) |
| `II` | 75 | 84 | adequate (>=75 each) |
| `III` | 75 | 75 | adequate (>=75 each) |
| `IV` | 75 | 75 | adequate (>=75 each) |
| `V` | 75 | 60 | corpus-limited cell (<75, never padded) |
| `VI` | 75 | 27 | corpus-limited cell (<75, never padded) |

| Modality | Benign (`locked_test_v3`) | Malignant (`locked_test_v3`) | Total |
|---|---|---|---|
| `dermoscopic` | 400 | 300 | 700 |
| `clinical` | 227 | 347 | 574 |
| `tbp_tile` | 250 | 100 | 350 |
