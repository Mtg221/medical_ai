# Model Weights License & External Data Access Notice (`DATA-002`, `REPRO-001`)

**Codebase License:** MIT License (see [LICENSE](LICENSE)).  
**Trained Model Weights License (`v0.2.0-research-artifacts`):** **Non-Commercial Research Only**

---

## 1. Non-Commercial Restriction on Released Weights (`DATA-002`)

While the source code in this repository is licensed under the MIT License, the released **trained model weights** (`models_nlp_18seed.tar.gz` and `models_vision_9seed.tar.gz`) were trained on a combination of open-license and non-commercial research corpora:

| Dataset | Track | License | Commercial Use |
|---|---|---|---|
| `SYNTH-MED-V3.1` | NLP | MIT (Repository) | Permitted |
| `MACCROBAT2020` | NLP | CC-BY-4.0 | Permitted with attribution |
| `PxCorpus` | NLP | CC-BY-SA-4.0 | ShareAlike required |
| `BC5CDR` / `NCBI Disease` | NLP | Public Domain | Permitted |
| `E3C Layer 1 & Layer 3` | NLP | **CC-BY-NC-4.0** | **Prohibited (Non-Commercial Only)** |
| `derm-corpus` (ISIC / HAM10000 / Clinical subsets) | Vision | **Mixed / CC-BY-NC** | **Prohibited (Non-Commercial Only)** |

**Therefore:** Any commercial deployment must retrain models strictly on commercially licensed datasets (`SYNTH-MED-V3.1`, `MACCROBAT2020`, `BC5CDR`, `NCBI Disease`, or institutional data with explicit rights).

---

## 2. Raw Dataset Access & Reproducibility Boundary (`REPRO-001`)

- **Fully self-contained on a fresh clone:** `SYNTH-MED-V3.1` generation (`medextract data build`), `hybrid_v2` CPU training, unit/API/Playwright E2E tests, and inference with downloaded release weights.
- **Requires external credentials:** Re-downloading raw `derm-corpus` skin thumbnails from scratch requires private CloudFront credentials (`DERM_CLOUDFRONT_USER` and `DERM_CLOUDFRONT_PASS`). External reviewers without those credentials can still verify all vision benchmarks and inference using the SHA-256-verified `models_vision_9seed.tar.gz` bundle and `scripts/package_release_artifacts.py unpack`.
