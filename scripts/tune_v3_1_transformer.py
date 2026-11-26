"""Tune the stacked multi-head pre-LayerNorm Transformer on SYNTH-MED-V3.1 val (600 train docs)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import yaml

from medresearch_platform.eval_harness.runner import run_experiment

TRIALS: list[dict[str, Any]] = [
    {"name": "L2_H4_D32_ep15", "num_layers": 2, "num_heads": 4, "hidden_dim": 32, "lr": 0.015, "epochs": 15, "patience": 5, "dropout": 0.10},
    {"name": "L2_H4_D48_ep18", "num_layers": 2, "num_heads": 4, "hidden_dim": 48, "lr": 0.012, "epochs": 18, "patience": 5, "dropout": 0.10},
    {"name": "L3_H4_D48_ep18", "num_layers": 3, "num_heads": 4, "hidden_dim": 48, "lr": 0.010, "epochs": 18, "patience": 5, "dropout": 0.10},
    {"name": "L2_H4_D64_ep18", "num_layers": 2, "num_heads": 4, "hidden_dim": 64, "lr": 0.010, "epochs": 18, "patience": 5, "dropout": 0.10},
]


def main() -> int:
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        for trial in TRIALS:
            cfg = {
                "approach": "transformer",
                "eval_split": "val",
                "data_dir": "data/processed/v3_1",
                "output_root": "experiments/runs/v3_1_transformer_tuning",
                "max_train_docs": None,
                "bpe_merges": 256,
                "seeds": [1],
                **{k: v for k, v in trial.items() if k != "name"},
            }
            p = tmp_dir / f"{trial['name']}.yaml"
            p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            res = run_experiment(p, seeds=[1])
            s = res["summary"]
            row = {
                "name": trial["name"],
                "config": {k: v for k, v in trial.items() if k != "name"},
                "strict_f1": round(float(s["strict_f1"]["mean"]), 4),
                "lenient_f1": round(float(s["lenient_f1"]["mean"]), 4),
                "wall_seconds": round(float(s["wall_seconds"]["mean"]), 2),
            }
            results.append(row)
            print(
                f"[tune] {row['name']:<16} | Strict={row['strict_f1']:.4f} "
                f"| Lenient={row['lenient_f1']:.4f} | wall={row['wall_seconds']:.1f}s",
                flush=True,
            )

    best = max(results, key=lambda r: r["strict_f1"])
    out = {
        "dataset": "SYNTH-MED-V3.1",
        "split": "val",
        "num_train_docs": 600,
        "seed": 1,
        "baseline_single_block_strict_f1": 0.5765,
        "trials": results,
        "best": best,
    }
    out_path = Path("docs/results/v3_1_transformer_tuning.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Best: {best['name']} (Strict={best['strict_f1']:.4f}) -> wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
