"""One-time locked test-split evaluation across all approaches on SYNTH-MED-V2 (seeds 1, 2, 3)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import yaml

from medresearch_platform.eval_harness.metrics import paired_significance_test
from medresearch_platform.eval_harness.runner import _load_jsonl, run_experiment
from medresearch_platform.approaches.neural_seq import predict_bilstm_crf
from medresearch_platform.hybrid.pipeline import predict_hybrid

CONFIG_NAMES = [
    "rule_based",
    "classical_ml",
    "classical_crf",
    "bilstm",
    "bilstm_crf",
    "transformer",
    "hybrid",
]


def main() -> int:
    results: dict[str, Any] = {"dataset": "SYNTH-MED-V2", "split": "test", "approaches": {}}
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        for name in CONFIG_NAMES:
            cfg = yaml.safe_load(Path(f"configs/exp/{name}.yaml").read_text(encoding="utf-8"))
            cfg["eval_split"] = "test"
            cfg["seeds"] = [1, 2, 3]
            tmp_cfg = tmp_dir / f"{name}_test.yaml"
            tmp_cfg.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            res = run_experiment(tmp_cfg, seeds=[1, 2, 3])
            results["approaches"][name] = {
                "run_id": res["run_id"],
                "summary": res["summary"],
                "error_counts": res["error_counts"],
                "per_seed": res["per_seed"],
            }
            sf1 = res["summary"]["strict_f1"]
            print(f"{name:<15} | Strict F1: {sf1['mean']:.4f} ± {sf1['std']:.4f}")

    # Paired significance: hybrid vs bilstm_crf on seed 1
    train_docs = _load_jsonl(Path("data/processed/v1/train.jsonl"))
    test_docs = _load_jsonl(Path("data/processed/v1/test.jsonl"))
    bilstm_cfg = yaml.safe_load(Path("configs/exp/bilstm_crf.yaml").read_text(encoding="utf-8"))
    hybrid_cfg = yaml.safe_load(Path("configs/exp/hybrid.yaml").read_text(encoding="utf-8"))
    preds_bilstm = predict_bilstm_crf(train_docs, test_docs, 1, bilstm_cfg)
    preds_hybrid = predict_hybrid(train_docs, test_docs, 1, hybrid_cfg)
    sig = paired_significance_test(test_docs, preds_bilstm, preds_hybrid, seed=42)
    results["significance_hybrid_vs_bilstm_crf"] = sig
    print("SIGNIFICANCE (hybrid vs bilstm_crf, seed=1):", sig)

    out_path = Path("docs/results/locked_test_evaluation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
