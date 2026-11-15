# Statistical Significance Tables — SYNTH-MED-V3.1 (val, n=160 docs, 5 seeds)

## 1. Per-Approach Performance (5-seed Mean ± 95% CI)

| Approach | Strict F1 [95% CI] | Lenient F1 [95% CI] | Assertion Macro-F1 | Relation F1 | TIMEX F1 | TIMEX Norm | Doc Acc |
|---|---|---|---|---|---|---|---|
| `rule_based` | 0.3148 [0.3148, 0.3148] | 0.3356 [0.3356, 0.3356] | 0.4354 [0.4354, 0.4354] | 0.0465 [0.0465, 0.0465] | 0.5073 | 0.1934 | 1.0000 |
| `classical_ml` | 0.9973 [0.9973, 0.9973] | 0.9993 [0.9993, 0.9993] | 0.7573 [0.7573, 0.7573] | 0.7192 [0.7192, 0.7192] | 0.7140 | 0.1184 | 1.0000 |
| `classical_crf` | 0.9961 [0.9961, 0.9961] | 0.9981 [0.9981, 0.9981] | 0.7567 [0.7567, 0.7567] | 0.7186 [0.7186, 0.7186] | 0.7140 | 0.1184 | 1.0000 |
| `bilstm` | 0.9665 [0.9628, 0.9702] | 0.9850 [0.9833, 0.9867] | 0.7548 [0.7513, 0.7582] | 0.7005 [0.6920, 0.7089] | 0.7009 | 0.1190 | 1.0000 |
| `bilstm_crf` | 0.9897 [0.9863, 0.9932] | 0.9933 [0.9900, 0.9966] | 0.7544 [0.7509, 0.7580] | 0.7155 [0.7127, 0.7182] | 0.7127 | 0.1176 | 1.0000 |
| `transformer` | 0.7852 [0.7640, 0.8064] | 0.8842 [0.8695, 0.8988] | 0.7499 [0.7428, 0.7571] | 0.5707 [0.5515, 0.5898] | 0.4740 | 0.1472 | 1.0000 |
| `hybrid` | 0.9942 [0.9913, 0.9972] | 0.9974 [0.9963, 0.9985] | 0.7350 [0.6488, 0.8212] | 0.8937 [0.8431, 0.9444] | 1.0000 | 1.0000 | 1.0000 |

*Method: 5-seed mean with normal 95% CI (`mean ± 1.96 * std / sqrt(5)`); pooled document-level bootstrap (10,000 resamples) CIs recorded in `docs/results/v3_1_val_significance.json`.*

## 2. All Pairwise Strict-F1 Comparisons (Holm-Bonferroni Corrected)

| Model A | Model B | F1 A | F1 B | ΔF1 (B - A) [95% CI] | Raw p | Holm-Bonferroni p | Sig (α=0.05) |
|---|---|---|---|---|---|---|---|
| `rule_based` | `classical_ml` | 0.3148 | 0.9973 | +0.6824 [+0.6559, +0.7101] | 0.0002 | 0.0042 | Yes |
| `rule_based` | `classical_crf` | 0.3148 | 0.9961 | +0.6812 [+0.6544, +0.7093] | 0.0002 | 0.0042 | Yes |
| `rule_based` | `bilstm` | 0.3148 | 0.9665 | +0.6517 [+0.6257, +0.6789] | 0.0002 | 0.0042 | Yes |
| `rule_based` | `bilstm_crf` | 0.3148 | 0.9897 | +0.6749 [+0.6483, +0.7022] | 0.0002 | 0.0042 | Yes |
| `rule_based` | `transformer` | 0.3148 | 0.7848 | +0.4700 [+0.4402, +0.5023] | 0.0002 | 0.0042 | Yes |
| `rule_based` | `hybrid` | 0.3148 | 0.9942 | +0.6794 [+0.6526, +0.7071] | 0.0002 | 0.0042 | Yes |
| `classical_ml` | `classical_crf` | 0.9973 | 0.9961 | -0.0012 [-0.0036, +0.0012] | 0.3086 | 0.3086 | No |
| `classical_ml` | `bilstm` | 0.9973 | 0.9665 | -0.0308 [-0.0351, -0.0266] | 0.0002 | 0.0042 | Yes |
| `classical_ml` | `bilstm_crf` | 0.9973 | 0.9897 | -0.0075 [-0.0103, -0.0050] | 0.0002 | 0.0042 | Yes |
| `classical_ml` | `transformer` | 0.9973 | 0.7848 | -0.2125 [-0.2299, -0.1948] | 0.0002 | 0.0042 | Yes |
| `classical_ml` | `hybrid` | 0.9973 | 0.9942 | -0.0031 [-0.0054, -0.0006] | 0.0198 | 0.0594 | No |
| `classical_crf` | `bilstm` | 0.9961 | 0.9665 | -0.0296 [-0.0340, -0.0254] | 0.0002 | 0.0042 | Yes |
| `classical_crf` | `bilstm_crf` | 0.9961 | 0.9897 | -0.0063 [-0.0091, -0.0036] | 0.0002 | 0.0042 | Yes |
| `classical_crf` | `transformer` | 0.9961 | 0.7848 | -0.2113 [-0.2287, -0.1935] | 0.0002 | 0.0042 | Yes |
| `classical_crf` | `hybrid` | 0.9961 | 0.9942 | -0.0019 [-0.0040, +0.0003] | 0.0866 | 0.1732 | No |
| `bilstm` | `bilstm_crf` | 0.9665 | 0.9897 | +0.0232 [+0.0198, +0.0268] | 0.0002 | 0.0042 | Yes |
| `bilstm` | `transformer` | 0.9665 | 0.7848 | -0.1817 [-0.1981, -0.1652] | 0.0002 | 0.0042 | Yes |
| `bilstm` | `hybrid` | 0.9665 | 0.9942 | +0.0277 [+0.0239, +0.0316] | 0.0002 | 0.0042 | Yes |
| `bilstm_crf` | `transformer` | 0.9897 | 0.7848 | -0.2049 [-0.2218, -0.1878] | 0.0002 | 0.0042 | Yes |
| `bilstm_crf` | `hybrid` | 0.9897 | 0.9942 | +0.0045 [+0.0027, +0.0063] | 0.0002 | 0.0042 | Yes |
| `transformer` | `hybrid` | 0.7848 | 0.9942 | +0.2094 [+0.1919, +0.2265] | 0.0002 | 0.0042 | Yes |

*Method: Cluster-paired bootstrap over 160 documents pooled across 5 seeds; 10,000 resamples; Holm-Bonferroni family-wise error correction across all 21 pairwise comparisons.*
