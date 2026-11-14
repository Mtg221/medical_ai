"""Run multi-seed SYNTH-MED-V3 validation benchmark and hybrid ablations (val split only)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import yaml

from medresearch_platform.eval_harness.runner import run_experiment

APPROACHES: tuple[str, ...] = (
    "rule_based",
    "classical_ml",
    "classical_crf",
    "bilstm",
    "bilstm_crf",
    "transformer",
    "hybrid",
)

ABLATIONS: tuple[tuple[str, dict[str, bool]], ...] = (
    ("full_hybrid", {}),
    ("no_transformer_fusion", {"use_transformer_fusion": False}),
    ("no_temporal_crf", {"use_temporal_crf": False}),
    ("no_relation_cnn", {"use_relation_cnn": False}),
    ("no_assertion_bilstm", {"use_assertion_bilstm": False}),
    ("no_rules_layer", {"use_rules_layer": False}),
)


def main() -> int:
    val_results: dict[str, Any] = {
        "dataset": "SYNTH-MED-V3",
        "split": "val",
        "num_docs": 160,
        "seeds": [1, 2, 3],
        "approaches": {},
    }
    for name in APPROACHES:
        cfg_path = Path(f"configs/exp/v3/{name}.yaml")
        res = run_experiment(cfg_path, seeds=[1, 2, 3])
        val_results["approaches"][name] = {
            "run_id": res["run_id"],
            "config_path": str(cfg_path),
            "summary": res["summary"],
            "error_counts": res["error_counts"],
            "per_seed": res["per_seed"],
        }
        sf1 = res["summary"]["strict_f1"]
        lf1 = res["summary"]["lenient_f1"]
        assert_acc = res["summary"].get("assertion_acc", {}).get("mean", 0.0)
        rel_f1 = res["summary"].get("relation_f1", {}).get("mean", 0.0)
        timex_f1 = res["summary"].get("timex_f1", {}).get("mean", 0.0)
        print(
            f"[val] {name:<14} | Strict={sf1['mean']:.4f}±{sf1['std']:.4f} "
            f"| Lenient={lf1['mean']:.4f} | Assert={assert_acc:.4f} "
            f"| Rel={rel_f1:.4f} | TIMEX={timex_f1:.4f}",
            flush=True,
        )

    out_val = Path("docs/results/v3_val_benchmark.json")
    out_val.parent.mkdir(parents=True, exist_ok=True)
    out_val.write_text(json.dumps(val_results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_val}", flush=True)

    # Hybrid ablations on val (seeds 1, 2, 3)
    base_hybrid = yaml.safe_load(Path("configs/exp/v3/hybrid.yaml").read_text(encoding="utf-8"))
    ablation_results: dict[str, Any] = {
        "dataset": "SYNTH-MED-V3",
        "split": "val",
        "num_docs": 160,
        "seeds": [1, 2, 3],
        "variants": {},
    }
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        for variant_name, overrides in ABLATIONS:
            if variant_name == "full_hybrid":
                ablation_results["variants"][variant_name] = val_results["approaches"]["hybrid"]
                continue
            cfg = dict(base_hybrid)
            cfg.update(overrides)
            cfg_file = tmp_dir / f"hybrid_{variant_name}.yaml"
            cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            res = run_experiment(cfg_file, seeds=[1, 2, 3])
            ablation_results["variants"][variant_name] = {
                "run_id": res["run_id"],
                "overrides": overrides,
                "summary": res["summary"],
                "error_counts": res["error_counts"],
                "per_seed": res["per_seed"],
            }
            sf1 = res["summary"]["strict_f1"]
            assert_acc = res["summary"].get("assertion_acc", {}).get("mean", 0.0)
            rel_f1 = res["summary"].get("relation_f1", {}).get("mean", 0.0)
            timex_f1 = res["summary"].get("timex_f1", {}).get("mean", 0.0)
            print(
                f"[ablation] {variant_name:<22} | Strict={sf1['mean']:.4f}±{sf1['std']:.4f} "
                f"| Assert={assert_acc:.4f} | Rel={rel_f1:.4f} | TIMEX={timex_f1:.4f}",
                flush=True,
            )

    out_abl = Path("docs/results/v3_val_ablation.json")
    out_abl.write_text(json.dumps(ablation_results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_abl}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
