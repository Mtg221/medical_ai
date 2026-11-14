# Phase 1 Baseline Experiments Report

- **Experiment ID:** `exp_phase1_baseline_v1`
- **Immutable Manifest Path:** `var/storage/experiments/exp_phase1_baseline_v1_manifest.json`
- **Manifest SHA-256:** `71f3be3ff95a4652233f1060347bf770349188ccc3957aa2f43da9933adf0432`
- **Hardware:** Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz (6C/12T), 32.0 GB RAM, macOS Darwin 25.6.0 (`x86_64`), CPU-only
- **Measured Resource Usage:** Runtime `1.2856 seconds`, Peak RSS `107.59 MB`

## 1. Key Experimental Results
1. **Approach 1 (Custom OCR — `CustomOCRConvNet`):**
   - Clean scans ($\sigma=0.01$): `CER = 0.0000`, `WER = 0.0000`
   - Perturbed noisy scans ($\sigma=0.14$): `CER = 0.1895`, `Char Accuracy = 0.8105`
2. **Approach 2 (Custom Bilingual Medical NLP — `CustomMedicalNLPNet`):**
   - Exact Span Precision: `0.9250`, Recall: `1.0000`, F1: `0.9610`
   - Assertion Accuracy: `0.9730` (95% Bootstrap CI `[0.9182, 1.0000]`)
   - Temporal Accuracy: `0.9730` (95% Bootstrap CI `[0.9189, 1.0000]`)
   - Calibration: `Brier Score = 0.0725`, `ECE = 0.1454`
3. **Approach 6 (Hybrid Retrieval — `CustomHybridRetrievalNet`):**
   - `MRR = 1.0000`, `nDCG@3 = 0.7654`, `Precision@3 = 0.6667`, `Recall@3 = 0.6667`
4. **Federated Multi-Node Simulation (`FEDPROX` + Byzantine Quarantine + DP):**
   - Participating sites: `inst_dakar_primary` (L2 norm `11.5321`), `inst_paris_partner` (L2 norm `11.4021`)
   - Compromised node quarantined automatically: `compromised_outlier_node` (L2 norm `288.3036`)
   - Global aggregated weights SHA-256: `7e5eadcfc90294d13f686e3e39b38ffbc5b86041226b5a7f7ab6be326f2dde18`
