# Dermatology Corpus Data Card (Sampling Plan v2)

- **Cross-split near-duplicates dropped from `locked_test_v2`:** `978`

## Split Summary (Real Images Only)

| Split | Total | Benign | Malignant | Unknown/Indet (Masked) | Clinical | Dermoscopic | TBP Tile |
|---|---|---|---|---|---|---|---|
| `train` | 23996 | 16098 | 5531 | 2367 | 4378 | 19618 | 0 |
| `val` | 3000 | 1860 | 918 | 222 | 653 | 2047 | 300 |
| `locked_test_v2` | 2022 | 789 | 1064 | 169 | 567 | 1161 | 294 |

## Fitzpatrick Skin Tone & Dataset Breakdown

| Split | Tone I | Tone II | Tone III | Tone IV | Tone V | Tone VI | UNKNOWN | Datasets |
|---|---|---|---|---|---|---|---|---|
| `train` | 816 | 3024 | 1363 | 622 | 523 | 451 | 17197 | `{'isic': 22628, 'skindisnet': 1368}` |
| `val` | 150 | 150 | 150 | 150 | 150 | 150 | 2100 | `{'isic': 2829, 'skindisnet': 171}` |
| `locked_test_v2` | 128 | 124 | 129 | 45 | 28 | 38 | 1530 | `{'isic': 1888, 'skindisnet': 134}` |
