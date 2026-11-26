"""One-time locked test-split evaluation across all approaches on SYNTH-MED-V3 (seeds 1, 2, 3)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import yaml

from medresearch_platform.approaches.classical_ml import predict_classical_crf
from medresearch_platform.approaches.neural_seq import predict_bilstm_crf
from medresearch_platform.eval_harness.metrics import paired_significance_test
from medresearch_platform.eval_harness.runner import _load_jsonl, run_experiment
from medresearch_platform.hybrid.pipeline import predict_hybrid

APPROACHES: tuple[str, ...] = (
    "rule_based",
    "classical_ml",
    "classical_crf",
    "bilstm",
    "bilstm_crf",
    "transformer",
    "hybrid",
)

LOCKED_TEST_SPLITS: tuple[str, ...] = (
    "test_iid",
    "test_unseen_terms",
    "test_unseen_templates",
)


def main() -> int:
    results: dict[str, Any] = {
        "dataset": "SYNTH-MED-V3",
        "seeds": [1, 2, 3],
        "splits": {},
    }
    train_docs = _load_jsonl(Path("data/processed/v3/train.jsonl"))
    crf_cfg = yaml.safe_load(Path("configs/exp/v3/classical_crf.yaml").read_text(encoding="utf-8"))
    bilstm_cfg = yaml.safe_load(Path("configs/exp/v3/bilstm_crf.yaml").read_text(encoding="utf-8"))
    hybrid_cfg = yaml.safe_load(Path("configs/exp/v3/hybrid.yaml").read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        for split in LOCKED_TEST_SPLITS:
            split_docs = _load_jsonl(Path(f"data/processed/v3/{split}.jsonl"))
            split_out: dict[str, Any] = {
                "num_docs": len(split_docs),
                "approaches": {},
                "significance": {},
            }
            for name in APPROACHES:
                cfg = yaml.safe_load(Path(f"configs/exp/v3/{name}.yaml").read_text(encoding="utf-8"))
                cfg["eval_split"] = split
                cfg["seeds"] = [1, 2, 3]
                tmp_cfg = tmp_dir / f"{name}_{split}.yaml"
                tmp_cfg.write_text(yaml.safe_dump(cfg), encoding="utf-8")
                res = run_experiment(tmp_cfg, seeds=[1, 2, 3])
                split_out["approaches"][name] = {
                    "run_id": res["run_id"],
                    "config_path": f"configs/exp/v3/{name}.yaml",
                    "summary": res["summary"],
                    "error_counts": res["error_counts"],
                    "per_seed": res["per_seed"],
                }
                sf1 = res["summary"]["strict_f1"]
                lf1 = res["summary"]["lenient_f1"]
                af1 = res["summary"]["assertion_macro_f1"]
                rf1 = res["summary"]["relations_f1"]
                tf1 = res["per_seed"][0]["metrics"]["timex"]["f1"]
                print(
                    f"[{split}] {name:<14} | Strict={sf1['mean']:.4f}±{sf1['std']:.4f} "
                    f"| Lenient={lf1['mean']:.4f} | AssertF1={af1['mean']:.4f} "
                    f"| RelF1={rf1['mean']:.4f} | TimexF1={tf1:.4f}",
                    flush=True,
                )

            preds_crf = predict_classical_crf(train_docs, split_docs, 1, crf_cfg)
            preds_bilstm = predict_bilstm_crf(train_docs, split_docs, 1, bilstm_cfg)
            preds_hybrid = predict_hybrid(train_docs, split_docs, 1, hybrid_cfg)
            split_out["significance"]["hybrid_vs_bilstm_crf"] = paired_significance_test(
                split_docs, preds_bilstm, preds_hybrid, seed=42
            )
            split_out["significance"]["bilstm_crf_vs_classical_crf"] = paired_significance_test(
                split_docs, preds_crf, preds_bilstm, seed=42
            )
            print(
                f"[{split}] SIGNIFICANCE hybrid vs bilstm_crf: "
                f"{split_out['significance']['hybrid_vs_bilstm_crf']}",
                flush=True,
            )
            results["splits"][split] = split_out

    out_path = Path("docs/results/v3_locked_test_evaluation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
