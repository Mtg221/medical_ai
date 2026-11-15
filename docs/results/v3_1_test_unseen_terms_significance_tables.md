# Statistical Significance Tables — SYNTH-MED-V3.1 (test_unseen_terms, n=160 docs, 5 seeds)

## 1. Per-Approach Performance (5-seed Mean ± 95% CI)

| Approach | Strict F1 [95% CI] | Lenient F1 [95% CI] | Assertion Macro-F1 | Relation F1 | TIMEX F1 | TIMEX Norm | Doc Acc |
|---|---|---|---|---|---|---|---|
| `rule_based` | 0.3233 [0.3233, 0.3233] | 0.3385 [0.3385, 0.3385] | 0.6602 [0.6602, 0.6602] | 0.0368 [0.0368, 0.0368] | 0.5096 | 0.2072 | 1.0000 |
| `classical_ml` | 0.8441 [0.8441, 0.8441] | 0.9353 [0.9353, 0.9353] | 0.7446 [0.7446, 0.7446] | 0.6701 [0.6701, 0.6701] | 0.7179 | 0.1265 | 1.0000 |
| `classical_crf` | 0.9169 [0.9169, 0.9169] | 0.9319 [0.9319, 0.9319] | 0.7511 [0.7511, 0.7511] | 0.7197 [0.7197, 0.7197] | 0.7179 | 0.1265 | 1.0000 |
| `bilstm` | 0.7859 [0.7770, 0.7949] | 0.8588 [0.8528, 0.8648] | 0.7411 [0.7281, 0.7541] | 0.6655 [0.6466, 0.6844] | 0.6994 | 0.1230 | 1.0000 |
| `bilstm_crf` | 0.8141 [0.8085, 0.8197] | 0.8416 [0.8315, 0.8517] | 0.7209 [0.7007, 0.7410] | 0.7018 [0.6909, 0.7128] | 0.7123 | 0.1222 | 1.0000 |
| `transformer` | 0.5241 [0.5150, 0.5333] | 0.6980 [0.6900, 0.7059] | 0.7075 [0.6898, 0.7252] | 0.3984 [0.3732, 0.4237] | 0.4505 | 0.1578 | 1.0000 |
| `hybrid` | 0.8277 [0.8072, 0.8482] | 0.8452 [0.8265, 0.8640] | 0.6672 [0.6020, 0.7324] | 0.8127 [0.7638, 0.8617] | 1.0000 | 1.0000 | 1.0000 |

*Method: 5-seed mean with normal 95% CI (`mean ± 1.96 * std / sqrt(5)`); pooled document-level bootstrap (10,000 resamples) CIs recorded in `docs/results/v3_1_test_unseen_terms_significance.json`.*

## 2. All Pairwise Strict-F1 Comparisons (Holm-Bonferroni Corrected)

| Model A | Model B | F1 A | F1 B | ΔF1 (B - A) [95% CI] | Raw p | Holm-Bonferroni p | Sig (α=0.05) |
|---|---|---|---|---|---|---|---|
| `rule_based` | `classical_ml` | 0.3233 | 0.8441 | +0.5208 [+0.4993, +0.5438] | 0.0002 | 0.0042 | Yes |
| `rule_based` | `classical_crf` | 0.3233 | 0.9169 | +0.5936 [+0.5720, +0.6167] | 0.0002 | 0.0042 | Yes |
| `rule_based` | `bilstm` | 0.3233 | 0.7859 | +0.4627 [+0.4446, +0.4817] | 0.0002 | 0.0042 | Yes |
| `rule_based` | `bilstm_crf` | 0.3233 | 0.8141 | +0.4909 [+0.4741, +0.5092] | 0.0002 | 0.0042 | Yes |
| `rule_based` | `transformer` | 0.3233 | 0.5241 | +0.2008 [+0.1768, +0.2270] | 0.0002 | 0.0042 | Yes |
| `rule_based` | `hybrid` | 0.3233 | 0.8278 | +0.5045 [+0.4867, +0.5240] | 0.0002 | 0.0042 | Yes |
| `classical_ml` | `classical_crf` | 0.8441 | 0.9169 | +0.0728 [+0.0595, +0.0863] | 0.0002 | 0.0042 | Yes |
| `classical_ml` | `bilstm` | 0.8441 | 0.7859 | -0.0582 [-0.0720, -0.0447] | 0.0002 | 0.0042 | Yes |
| `classical_ml` | `bilstm_crf` | 0.8441 | 0.8141 | -0.0300 [-0.0437, -0.0170] | 0.0002 | 0.0042 | Yes |
| `classical_ml` | `transformer` | 0.8441 | 0.5241 | -0.3201 [-0.3386, -0.3018] | 0.0002 | 0.0042 | Yes |
| `classical_ml` | `hybrid` | 0.8441 | 0.8278 | -0.0163 [-0.0300, -0.0032] | 0.0146 | 0.0146 | Yes |
| `classical_crf` | `bilstm` | 0.9169 | 0.7859 | -0.1310 [-0.1471, -0.1151] | 0.0002 | 0.0042 | Yes |
| `classical_crf` | `bilstm_crf` | 0.9169 | 0.8141 | -0.1028 [-0.1173, -0.0890] | 0.0002 | 0.0042 | Yes |
| `classical_crf` | `transformer` | 0.9169 | 0.5241 | -0.3928 [-0.4140, -0.3714] | 0.0002 | 0.0042 | Yes |
| `classical_crf` | `hybrid` | 0.9169 | 0.8278 | -0.0891 [-0.1035, -0.0753] | 0.0002 | 0.0042 | Yes |
| `bilstm` | `bilstm_crf` | 0.7859 | 0.8141 | +0.0282 [+0.0204, +0.0360] | 0.0002 | 0.0042 | Yes |
| `bilstm` | `transformer` | 0.7859 | 0.5241 | -0.2619 [-0.2801, -0.2436] | 0.0002 | 0.0042 | Yes |
| `bilstm` | `hybrid` | 0.7859 | 0.8278 | +0.0418 [+0.0329, +0.0506] | 0.0002 | 0.0042 | Yes |
| `bilstm_crf` | `transformer` | 0.8141 | 0.5241 | -0.2901 [-0.3065, -0.2727] | 0.0002 | 0.0042 | Yes |
| `bilstm_crf` | `hybrid` | 0.8141 | 0.8278 | +0.0136 [+0.0082, +0.0192] | 0.0002 | 0.0042 | Yes |
| `transformer` | `hybrid` | 0.5241 | 0.8278 | +0.3037 [+0.2866, +0.3199] | 0.0002 | 0.0042 | Yes |

*Method: Cluster-paired bootstrap over 160 documents pooled across 5 seeds; 10,000 resamples; Holm-Bonferroni family-wise error correction across all 21 pairwise comparisons.*
