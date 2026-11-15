"""Generate 5-seed summary CIs and Holm-Bonferroni corrected pairwise significance tables from JSON."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from medresearch_platform.eval_harness.metrics import holm_bonferroni_adjust, pooled_paired_bootstrap

METRIC_KEYS: tuple[str, ...] = (
    "strict_f1",
    "lenient_f1",
    "assertion_macro_f1",
    "relation_f1",
    "timex_f1",
    "timex_norm_acc",
    "doc_type_acc",
)


def _format_p(p: float) -> str:
    if p < 0.0001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def generate_tables(
    input_json: Path,
    output_json: Path,
    output_md: Path,
    *,
    n_resamples: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    raw = json.loads(input_json.read_text(encoding="utf-8"))
    approaches = list(raw["approaches"].keys())

    # 1. Per-approach summary across seeds + pooled bootstrap CI
    summary_table: dict[str, Any] = {}
    for app in approaches:
        per_seed = raw["approaches"][app]["per_seed"]
        n_seeds = len(per_seed)
        metrics_summary: dict[str, dict[str, float]] = {}
        for k in METRIC_KEYS:
            vals = [float(r[k]) for r in per_seed]
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if n_seeds > 1 else 0.0
            ci = 1.96 * std / math.sqrt(n_seeds) if n_seeds > 1 else 0.0
            metrics_summary[k] = {
                "mean": round(mean, 4),
                "std": round(std, 4),
                "ci95_low": round(max(0.0, mean - ci), 4),
                "ci95_high": round(min(1.0, mean + ci), 4),
            }
        # Pooled document-level bootstrap CI on strict F1 across all seeds
        counts = raw["approaches"][app]["doc_seed_counts"]
        arr = np.asarray(counts, dtype=np.float64).sum(axis=1)
        rng = np.random.default_rng(seed)
        n_docs = len(arr)
        idx = rng.integers(0, n_docs, size=(n_resamples, n_docs))
        samp = arr[idx].sum(axis=1)
        denom = 2.0 * samp[:, 0] + samp[:, 1] + samp[:, 2]
        f1s = np.sort(np.where(denom > 0.0, (2.0 * samp[:, 0]) / np.where(denom > 0.0, denom, 1.0), 0.0))
        metrics_summary["strict_f1_pooled_boot_ci95"] = {
            "mean": round(float(f1s.mean()), 4),
            "ci95_low": round(float(f1s[max(0, int(0.025 * n_resamples))]), 4),
            "ci95_high": round(float(f1s[min(n_resamples - 1, int(0.975 * n_resamples))]), 4),
        }
        summary_table[app] = metrics_summary

    # 2. All pairwise comparisons + Holm-Bonferroni adjustment
    pairs = list(itertools.combinations(approaches, 2))
    pairwise_rows: list[dict[str, Any]] = []
    raw_pvals: list[float] = []
    for a, b in pairs:
        ca = [[(int(c[0]), int(c[1]), int(c[2])) for c in doc_seeds] for doc_seeds in raw["approaches"][a]["doc_seed_counts"]]
        cb = [[(int(c[0]), int(c[1]), int(c[2])) for c in doc_seeds] for doc_seeds in raw["approaches"][b]["doc_seed_counts"]]
        res = pooled_paired_bootstrap(ca, cb, n_resamples=n_resamples, seed=seed)
        raw_pvals.append(float(res["p_value"]))
        pairwise_rows.append(
            {
                "model_a": a,
                "model_b": b,
                "f1_a": res["f1_a"],
                "f1_b": res["f1_b"],
                "delta_f1": res["delta_f1"],
                "ci95_low": res["ci95_low"],
                "ci95_high": res["ci95_high"],
                "raw_p_value": res["p_value"],
            }
        )

    adj_pvals = holm_bonferroni_adjust(raw_pvals)
    for row, adj in zip(pairwise_rows, adj_pvals):
        row["holm_p_value"] = round(float(adj), 8)
        row["significant_alpha_0_05"] = bool(adj < 0.05)

    out_payload = {
        "dataset": raw["dataset"],
        "split": raw["split"],
        "num_eval_docs": raw["num_eval_docs"],
        "num_seeds": len(raw["approaches"][approaches[0]]["seeds"]),
        "method": "Cluster-paired document bootstrap pooled across 5 seeds",
        "n_resamples": n_resamples,
        "correction": f"Holm-Bonferroni across all {len(pairwise_rows)} pairwise comparisons",
        "summary_table": summary_table,
        "pairwise_comparisons": pairwise_rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")

    # 3. Render Markdown tables
    lines = [
        f"# Statistical Significance Tables — {raw['dataset']} ({raw['split']}, n={raw['num_eval_docs']} docs, 5 seeds)",
        "",
        "## 1. Per-Approach Performance (5-seed Mean ± 95% CI)",
        "",
        "| Approach | Strict F1 [95% CI] | Lenient F1 [95% CI] | Assertion Macro-F1 | Relation F1 | TIMEX F1 | TIMEX Norm | Doc Acc |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for app in approaches:
        ms = summary_table[app]
        sf = ms["strict_f1"]
        lf = ms["lenient_f1"]
        af = ms["assertion_macro_f1"]
        rf = ms["relation_f1"]
        tf = ms["timex_f1"]
        tn = ms["timex_norm_acc"]
        da = ms["doc_type_acc"]
        lines.append(
            f"| `{app}` | {sf['mean']:.4f} [{sf['ci95_low']:.4f}, {sf['ci95_high']:.4f}] "
            f"| {lf['mean']:.4f} [{lf['ci95_low']:.4f}, {lf['ci95_high']:.4f}] "
            f"| {af['mean']:.4f} [{af['ci95_low']:.4f}, {af['ci95_high']:.4f}] "
            f"| {rf['mean']:.4f} [{rf['ci95_low']:.4f}, {rf['ci95_high']:.4f}] "
            f"| {tf['mean']:.4f} | {tn['mean']:.4f} | {da['mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"*Method: 5-seed mean with normal 95% CI (`mean ± 1.96 * std / sqrt(5)`); pooled document-level bootstrap ({n_resamples:,} resamples) CIs recorded in `{output_json}`.*",
            "",
            "## 2. All Pairwise Strict-F1 Comparisons (Holm-Bonferroni Corrected)",
            "",
            "| Model A | Model B | F1 A | F1 B | ΔF1 (B - A) [95% CI] | Raw p | Holm-Bonferroni p | Sig (α=0.05) |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in pairwise_rows:
        sig = "Yes" if row["significant_alpha_0_05"] else "No"
        lines.append(
            f"| `{row['model_a']}` | `{row['model_b']}` | {row['f1_a']:.4f} | {row['f1_b']:.4f} "
            f"| {row['delta_f1']:+.4f} [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] "
            f"| {_format_p(row['raw_p_value'])} | {_format_p(row['holm_p_value'])} | {sig} |"
        )
    lines.extend(
        [
            "",
            f"*Method: Cluster-paired bootstrap over {raw['num_eval_docs']} documents pooled across 5 seeds; {n_resamples:,} resamples; Holm-Bonferroni family-wise error correction across all {len(pairwise_rows)} pairwise comparisons.*",
            "",
        ]
    )
    output_md.write_text("\n".join(lines), encoding="utf-8")
    return out_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("docs/results/v3_1_val_5seed_benchmark.json"))
    parser.add_argument("--output-json", type=Path, default=Path("docs/results/v3_1_val_significance.json"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/results/v3_1_val_significance_tables.md"))
    parser.add_argument("--resamples", type=int, default=10000)
    args = parser.parse_args()
    out = generate_tables(args.input, args.output_json, args.output_md, n_resamples=args.resamples)
    print(
        f"Generated {len(out['pairwise_comparisons'])} pairwise comparisons "
        f"({out['n_resamples']} resamples, {out['correction']}) -> {args.output_md}"
    )


if __name__ == "__main__":
    main()
