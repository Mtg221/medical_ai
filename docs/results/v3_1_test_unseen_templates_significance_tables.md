# Statistical Significance Tables — SYNTH-MED-V3.1 (test_unseen_templates, n=160 docs, 5 seeds)

## 1. Per-Approach Performance (5-seed Mean ± 95% CI)

| Approach | Strict F1 [95% CI] | Lenient F1 [95% CI] | Assertion Macro-F1 | Relation F1 | TIMEX F1 | TIMEX Norm | Doc Acc |
|---|---|---|---|---|---|---|---|
| `rule_based` | 0.3066 [0.3066, 0.3066] | 0.3165 [0.3165, 0.3165] | 0.5430 [0.5430, 0.5430] | 0.0506 [0.0506, 0.0506] | 0.5029 | 0.1781 | 1.0000 |
| `classical_ml` | 0.9670 [0.9670, 0.9670] | 0.9801 [0.9801, 0.9801] | 0.6453 [0.6453, 0.6453] | 0.7105 [0.7105, 0.7105] | 0.7340 | 0.1032 | 1.0000 |
| `classical_crf` | 0.9527 [0.9527, 0.9527] | 0.9701 [0.9701, 0.9701] | 0.6459 [0.6459, 0.6459] | 0.7235 [0.7235, 0.7235] | 0.7340 | 0.1032 | 1.0000 |
| `bilstm` | 0.8619 [0.8501, 0.8737] | 0.9226 [0.9145, 0.9307] | 0.6440 [0.6371, 0.6508] | 0.6453 [0.6280, 0.6625] | 0.7154 | 0.0999 | 1.0000 |
| `bilstm_crf` | 0.9392 [0.9309, 0.9474] | 0.9721 [0.9659, 0.9784] | 0.6399 [0.6372, 0.6426] | 0.6999 [0.6973, 0.7025] | 0.7309 | 0.0994 | 1.0000 |
| `transformer` | 0.6886 [0.6756, 0.7016] | 0.7816 [0.7678, 0.7955] | 0.6462 [0.6419, 0.6505] | 0.5255 [0.5062, 0.5448] | 0.4620 | 0.1271 | 1.0000 |
| `hybrid` | 0.8949 [0.8796, 0.9102] | 0.9644 [0.9589, 0.9698] | 0.5283 [0.4996, 0.5569] | 0.8029 [0.7490, 0.8568] | 0.9969 | 1.0000 | 1.0000 |

*Method: 5-seed mean with normal 95% CI (`mean ± 1.96 * std / sqrt(5)`); pooled document-level bootstrap (10,000 resamples) CIs recorded in `docs/results/v3_1_test_unseen_templates_significance.json`.*

## 2. All Pairwise Strict-F1 Comparisons (Holm-Bonferroni Corrected)

| Model A | Model B | F1 A | F1 B | ΔF1 (B - A) [95% CI] | Raw p | Holm-Bonferroni p | Sig (α=0.05) |
|---|---|---|---|---|---|---|---|
| `rule_based` | `classical_ml` | 0.3066 | 0.9670 | +0.6604 [+0.6365, +0.6850] | 0.0002 | 0.0042 | Yes |
| `rule_based` | `classical_crf` | 0.3066 | 0.9527 | +0.6461 [+0.6225, +0.6706] | 0.0002 | 0.0042 | Yes |
| `rule_based` | `bilstm` | 0.3066 | 0.8618 | +0.5552 [+0.5334, +0.5777] | 0.0002 | 0.0042 | Yes |
| `rule_based` | `bilstm_crf` | 0.3066 | 0.9391 | +0.6326 [+0.6096, +0.6565] | 0.0002 | 0.0042 | Yes |
| `rule_based` | `transformer` | 0.3066 | 0.6884 | +0.3818 [+0.3567, +0.4085] | 0.0002 | 0.0042 | Yes |
| `rule_based` | `hybrid` | 0.3066 | 0.8948 | +0.5882 [+0.5666, +0.6107] | 0.0002 | 0.0042 | Yes |
| `classical_ml` | `classical_crf` | 0.9670 | 0.9527 | -0.0143 [-0.0207, -0.0081] | 0.0002 | 0.0042 | Yes |
| `classical_ml` | `bilstm` | 0.9670 | 0.8618 | -0.1052 [-0.1154, -0.0949] | 0.0002 | 0.0042 | Yes |
| `classical_ml` | `bilstm_crf` | 0.9670 | 0.9391 | -0.0279 [-0.0345, -0.0212] | 0.0002 | 0.0042 | Yes |
| `classical_ml` | `transformer` | 0.9670 | 0.6884 | -0.2786 [-0.2942, -0.2631] | 0.0002 | 0.0042 | Yes |
| `classical_ml` | `hybrid` | 0.9670 | 0.8948 | -0.0722 [-0.0815, -0.0629] | 0.0002 | 0.0042 | Yes |
| `classical_crf` | `bilstm` | 0.9527 | 0.8618 | -0.0909 [-0.1010, -0.0808] | 0.0002 | 0.0042 | Yes |
| `classical_crf` | `bilstm_crf` | 0.9527 | 0.9391 | -0.0136 [-0.0209, -0.0060] | 0.0004 | 0.0042 | Yes |
| `classical_crf` | `transformer` | 0.9527 | 0.6884 | -0.2643 [-0.2796, -0.2494] | 0.0002 | 0.0042 | Yes |
| `classical_crf` | `hybrid` | 0.9527 | 0.8948 | -0.0579 [-0.0665, -0.0494] | 0.0002 | 0.0042 | Yes |
| `bilstm` | `bilstm_crf` | 0.8618 | 0.9391 | +0.0773 [+0.0697, +0.0851] | 0.0002 | 0.0042 | Yes |
| `bilstm` | `transformer` | 0.8618 | 0.6884 | -0.1734 [-0.1869, -0.1600] | 0.0002 | 0.0042 | Yes |
| `bilstm` | `hybrid` | 0.8618 | 0.8948 | +0.0330 [+0.0270, +0.0390] | 0.0002 | 0.0042 | Yes |
| `bilstm_crf` | `transformer` | 0.9391 | 0.6884 | -0.2507 [-0.2654, -0.2360] | 0.0002 | 0.0042 | Yes |
| `bilstm_crf` | `hybrid` | 0.9391 | 0.8948 | -0.0443 [-0.0507, -0.0384] | 0.0002 | 0.0042 | Yes |
| `transformer` | `hybrid` | 0.6884 | 0.8948 | +0.2064 [+0.1922, +0.2202] | 0.0002 | 0.0042 | Yes |

*Method: Cluster-paired bootstrap over 160 documents pooled across 5 seeds; 10,000 resamples; Holm-Bonferroni family-wise error correction across all 21 pairwise comparisons.*
