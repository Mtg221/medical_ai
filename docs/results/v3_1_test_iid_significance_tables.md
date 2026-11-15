# Statistical Significance Tables — SYNTH-MED-V3.1 (test_iid, n=160 docs, 5 seeds)

## 1. Per-Approach Performance (5-seed Mean ± 95% CI)

| Approach | Strict F1 [95% CI] | Lenient F1 [95% CI] | Assertion Macro-F1 | Relation F1 | TIMEX F1 | TIMEX Norm | Doc Acc |
|---|---|---|---|---|---|---|---|
| `rule_based` | 0.3221 [0.3221, 0.3221] | 0.3343 [0.3343, 0.3343] | 0.5878 [0.5878, 0.5878] | 0.0425 [0.0425, 0.0425] | 0.5101 | 0.1984 | 1.0000 |
| `classical_ml` | 0.9983 [0.9983, 0.9983] | 0.9993 [0.9993, 0.9993] | 0.7138 [0.7138, 0.7138] | 0.7340 [0.7340, 0.7340] | 0.7160 | 0.1217 | 1.0000 |
| `classical_crf` | 0.9968 [0.9968, 0.9968] | 0.9982 [0.9982, 0.9982] | 0.7194 [0.7194, 0.7194] | 0.7341 [0.7341, 0.7341] | 0.7167 | 0.1217 | 1.0000 |
| `bilstm` | 0.9659 [0.9627, 0.9691] | 0.9821 [0.9803, 0.9839] | 0.7198 [0.7143, 0.7252] | 0.7208 [0.7155, 0.7261] | 0.6997 | 0.1198 | 1.0000 |
| `bilstm_crf` | 0.9884 [0.9859, 0.9909] | 0.9928 [0.9906, 0.9950] | 0.7168 [0.7150, 0.7186] | 0.7342 [0.7327, 0.7357] | 0.7155 | 0.1204 | 1.0000 |
| `transformer` | 0.7930 [0.7781, 0.8080] | 0.8857 [0.8745, 0.8969] | 0.7305 [0.7286, 0.7323] | 0.5978 [0.5834, 0.6122] | 0.4557 | 0.1549 | 1.0000 |
| `hybrid` | 0.9941 [0.9912, 0.9969] | 0.9974 [0.9962, 0.9986] | 0.7325 [0.6503, 0.8147] | 0.8922 [0.8415, 0.9429] | 0.9986 | 1.0000 | 1.0000 |

*Method: 5-seed mean with normal 95% CI (`mean ± 1.96 * std / sqrt(5)`); pooled document-level bootstrap (10,000 resamples) CIs recorded in `docs/results/v3_1_test_iid_significance.json`.*

## 2. All Pairwise Strict-F1 Comparisons (Holm-Bonferroni Corrected)

| Model A | Model B | F1 A | F1 B | ΔF1 (B - A) [95% CI] | Raw p | Holm-Bonferroni p | Sig (α=0.05) |
|---|---|---|---|---|---|---|---|
| `rule_based` | `classical_ml` | 0.3221 | 0.9983 | +0.6762 [+0.6509, +0.7019] | 0.0002 | 0.0042 | Yes |
| `rule_based` | `classical_crf` | 0.3221 | 0.9968 | +0.6747 [+0.6493, +0.7006] | 0.0002 | 0.0042 | Yes |
| `rule_based` | `bilstm` | 0.3221 | 0.9659 | +0.6438 [+0.6198, +0.6687] | 0.0002 | 0.0042 | Yes |
| `rule_based` | `bilstm_crf` | 0.3221 | 0.9884 | +0.6662 [+0.6417, +0.6915] | 0.0002 | 0.0042 | Yes |
| `rule_based` | `transformer` | 0.3221 | 0.7929 | +0.4707 [+0.4428, +0.5008] | 0.0002 | 0.0042 | Yes |
| `rule_based` | `hybrid` | 0.3221 | 0.9941 | +0.6719 [+0.6470, +0.6977] | 0.0002 | 0.0042 | Yes |
| `classical_ml` | `classical_crf` | 0.9983 | 0.9968 | -0.0015 [-0.0037, +0.0006] | 0.1464 | 0.1464 | No |
| `classical_ml` | `bilstm` | 0.9983 | 0.9659 | -0.0324 [-0.0369, -0.0280] | 0.0002 | 0.0042 | Yes |
| `classical_ml` | `bilstm_crf` | 0.9983 | 0.9884 | -0.0100 [-0.0124, -0.0074] | 0.0002 | 0.0042 | Yes |
| `classical_ml` | `transformer` | 0.9983 | 0.7929 | -0.2055 [-0.2223, -0.1883] | 0.0002 | 0.0042 | Yes |
| `classical_ml` | `hybrid` | 0.9983 | 0.9941 | -0.0043 [-0.0066, -0.0019] | 0.0008 | 0.0042 | Yes |
| `classical_crf` | `bilstm` | 0.9968 | 0.9659 | -0.0309 [-0.0355, -0.0264] | 0.0002 | 0.0042 | Yes |
| `classical_crf` | `bilstm_crf` | 0.9968 | 0.9884 | -0.0084 [-0.0112, -0.0057] | 0.0002 | 0.0042 | Yes |
| `classical_crf` | `transformer` | 0.9968 | 0.7929 | -0.2040 [-0.2205, -0.1874] | 0.0002 | 0.0042 | Yes |
| `classical_crf` | `hybrid` | 0.9968 | 0.9941 | -0.0028 [-0.0049, -0.0007] | 0.0078 | 0.0156 | Yes |
| `bilstm` | `bilstm_crf` | 0.9659 | 0.9884 | +0.0225 [+0.0191, +0.0261] | 0.0002 | 0.0042 | Yes |
| `bilstm` | `transformer` | 0.9659 | 0.7929 | -0.1730 [-0.1884, -0.1572] | 0.0002 | 0.0042 | Yes |
| `bilstm` | `hybrid` | 0.9659 | 0.9941 | +0.0282 [+0.0243, +0.0322] | 0.0002 | 0.0042 | Yes |
| `bilstm_crf` | `transformer` | 0.9884 | 0.7929 | -0.1955 [-0.2119, -0.1790] | 0.0002 | 0.0042 | Yes |
| `bilstm_crf` | `hybrid` | 0.9884 | 0.9941 | +0.0057 [+0.0038, +0.0075] | 0.0002 | 0.0042 | Yes |
| `transformer` | `hybrid` | 0.7929 | 0.9941 | +0.2012 [+0.1846, +0.2176] | 0.0002 | 0.0042 | Yes |

*Method: Cluster-paired bootstrap over 160 documents pooled across 5 seeds; 10,000 resamples; Holm-Bonferroni family-wise error correction across all 21 pairwise comparisons.*
