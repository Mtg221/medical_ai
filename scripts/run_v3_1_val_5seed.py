"""Run all 7 final V3.1 configs across 5 seeds on val and record per-doc (tp, fp, fn) counts."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import yaml

from medresearch_platform.data.schema import AnnotatedDocument
from medresearch_platform.eval_harness.metrics import Evaluator, per_doc_strict_counts
from medresearch_platform.eval_harness.profiler import ResourceProfiler
from medresearch_platform.eval_harness.runner import APPROACH_REGISTRY

APPROACHES: tuple[str, ...] = (
    "rule_based",
    "classical_ml",
    "classical_crf",
    "bilstm",
    "bilstm_crf",
    "transformer",
    "hybrid",
)


def _load_split(path: Path) -> list[AnnotatedDocument]:
    with path.open("r", encoding="utf-8") as fh:
        return [AnnotatedDocument.from_dict(json.loads(line)) for line in fh if line.strip()]


def main() -> None:
    t_start = time.perf_counter()
    data_dir = Path("data/processed/v3_1")
    train_docs = _load_split(data_dir / "train.jsonl")
    val_docs = _load_split(data_dir / "val.jsonl")
    evaluator = Evaluator()

    results: dict[str, Any] = {
        "dataset": "SYNTH-MED-V3.1",
        "split": "val",
        "num_train_docs": len(train_docs),
        "num_eval_docs": len(val_docs),
        "approaches": {},
    }

    for approach in APPROACHES:
        cfg = yaml.safe_load(Path(f"configs/exp/v3_1/{approach}.yaml").read_text(encoding="utf-8"))
        seeds = [int(s) for s in cfg.get("seeds", [1, 2, 3, 4, 5])]
        fn = APPROACH_REGISTRY[approach]

        per_seed_metrics: list[dict[str, Any]] = []
        # N_docs x N_seeds of [tp, fp, fn]
        doc_seed_counts: list[list[list[int]]] = [[] for _ in range(len(val_docs))]

        for seed in seeds:
            with ResourceProfiler(docs_count=len(val_docs)) as prof:
                preds = fn(train_docs, val_docs, seed, cfg)
            ev = evaluator.evaluate(val_docs, preds)
            res_dict = prof.metrics.to_dict() if prof.metrics else {}
            counts = per_doc_strict_counts(val_docs, preds)
            for idx, (tp, fp, fn_c) in enumerate(counts):
                doc_seed_counts[idx].append([int(tp), int(fp), int(fn_c)])

            row = {
                "seed": seed,
                "strict_f1": float(ev["strict_span"]["f1"]),
                "lenient_f1": float(ev["lenient_span"]["f1"]),
                "assertion_macro_f1": float(ev["assertion"]["macro_f1"]),
                "relation_f1": float(ev["relations"]["f1"]),
                "timex_f1": float(ev["timex"]["f1"]),
                "timex_norm_acc": float(ev["timex"]["normalization_accuracy"]),
                "doc_type_acc": float(ev["doc_classification"]["accuracy"]),
                "wall_seconds": float(res_dict.get("wall_seconds", 0.0)),
                "peak_rss_mb": float(res_dict.get("peak_rss_mb", 0.0)),
            }
            per_seed_metrics.append(row)
            print(
                f"[{approach}][seed={seed}] strict={row['strict_f1']:.4f} "
                f"assert={row['assertion_macro_f1']:.4f} rel={row['relation_f1']:.4f} "
                f"timex={row['timex_f1']:.4f} norm={row['timex_norm_acc']:.4f} "
                f"({row['wall_seconds']:.1f}s)",
                flush=True,
            )

        results["approaches"][approach] = {
            "seeds": seeds,
            "per_seed": per_seed_metrics,
            "doc_seed_counts": doc_seed_counts,
        }

    results["total_runtime_s"] = round(time.perf_counter() - t_start, 2)
    out_path = Path("docs/results/v3_1_val_5seed_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved 5-seed val benchmark to {out_path} in {results['total_runtime_s']}s")


if __name__ == "__main__":
    main()
