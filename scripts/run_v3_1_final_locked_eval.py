"""Run all 7 final V3.1 configs across 5 seeds on val, test_iid, test_unseen_terms, and test_unseen_templates."""

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
from scripts.generate_significance_tables import generate_tables

APPROACHES: tuple[str, ...] = (
    "rule_based",
    "classical_ml",
    "classical_crf",
    "bilstm",
    "bilstm_crf",
    "transformer",
    "hybrid",
)
SPLITS: tuple[str, ...] = ("val", "test_iid", "test_unseen_terms", "test_unseen_templates")


def _load_split(path: Path) -> list[AnnotatedDocument]:
    with path.open("r", encoding="utf-8") as fh:
        return [AnnotatedDocument.from_dict(json.loads(line)) for line in fh if line.strip()]


def main() -> None:
    t_start = time.perf_counter()
    data_dir = Path("data/processed/v3_1")
    train_docs = _load_split(data_dir / "train.jsonl")
    split_docs = {s: _load_split(data_dir / f"{s}.jsonl") for s in SPLITS}
    all_eval_docs = [d for s in SPLITS for d in split_docs[s]]

    split_offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for s in SPLITS:
        n = len(split_docs[s])
        split_offsets[s] = (cursor, cursor + n)
        cursor += n

    evaluator = Evaluator()
    per_split_payloads: dict[str, dict[str, Any]] = {
        s: {
            "dataset": "SYNTH-MED-V3.1",
            "split": s,
            "num_train_docs": len(train_docs),
            "num_eval_docs": len(split_docs[s]),
            "approaches": {},
        }
        for s in SPLITS
    }

    for approach in APPROACHES:
        cfg = yaml.safe_load(Path(f"configs/exp/v3_1/{approach}.yaml").read_text(encoding="utf-8"))
        seeds = [int(s) for s in cfg.get("seeds", [1, 2, 3, 4, 5])]
        fn = APPROACH_REGISTRY[approach]

        for s in SPLITS:
            per_split_payloads[s]["approaches"][approach] = {
                "seeds": seeds,
                "per_seed": [],
                "doc_seed_counts": [[] for _ in range(len(split_docs[s]))],
            }

        for seed in seeds:
            with ResourceProfiler(docs_count=len(all_eval_docs)) as prof:
                preds_all = fn(train_docs, all_eval_docs, seed, cfg)
            res_dict = prof.metrics.to_dict() if prof.metrics else {}
            wall_s = float(res_dict.get("wall_seconds", 0.0))
            rss_mb = float(res_dict.get("peak_rss_mb", 0.0))

            for s in SPLITS:
                lo, hi = split_offsets[s]
                s_docs = split_docs[s]
                s_preds = preds_all[lo:hi]
                ev = evaluator.evaluate(s_docs, s_preds)
                counts = per_doc_strict_counts(s_docs, s_preds)
                for idx, (tp, fp, fn_c) in enumerate(counts):
                    per_split_payloads[s]["approaches"][approach]["doc_seed_counts"][idx].append(
                        [int(tp), int(fp), int(fn_c)]
                    )
                per_lang = ev.get("per_language", {})
                row = {
                    "seed": seed,
                    "strict_f1": float(ev["strict_span"]["f1"]),
                    "lenient_f1": float(ev["lenient_span"]["f1"]),
                    "en_strict_f1": float(per_lang.get("en", {}).get("f1", 0.0)),
                    "fr_strict_f1": float(per_lang.get("fr", {}).get("f1", 0.0)),
                    "assertion_macro_f1": float(ev["assertion"]["macro_f1"]),
                    "relation_f1": float(ev["relations"]["f1"]),
                    "timex_f1": float(ev["timex"]["f1"]),
                    "timex_norm_acc": float(ev["timex"]["normalization_accuracy"]),
                    "doc_type_acc": float(ev["doc_classification"]["accuracy"]),
                    "ece": float(ev["calibration"]["ece"]),
                    "wall_seconds": wall_s,
                    "peak_rss_mb": rss_mb,
                }
                per_split_payloads[s]["approaches"][approach]["per_seed"].append(row)

            iid_row = per_split_payloads["test_iid"]["approaches"][approach]["per_seed"][-1]
            ut_row = per_split_payloads["test_unseen_terms"]["approaches"][approach]["per_seed"][-1]
            utm_row = per_split_payloads["test_unseen_templates"]["approaches"][approach]["per_seed"][-1]
            print(
                f"[{approach}][seed={seed}] ({wall_s:.1f}s, {rss_mb:.0f}MB) | "
                f"iid={iid_row['strict_f1']:.4f} | unseen_terms={ut_row['strict_f1']:.4f} | "
                f"unseen_tpl={utm_row['strict_f1']:.4f}",
                flush=True,
            )

    out_dir = Path("docs/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    sig_summaries: dict[str, Any] = {}
    for s in SPLITS:
        split_json = out_dir / f"v3_1_{s}_5seed_benchmark.json"
        split_json.write_text(json.dumps(per_split_payloads[s], indent=2), encoding="utf-8")
        sig_summaries[s] = generate_tables(
            split_json,
            out_dir / f"v3_1_{s}_significance.json",
            out_dir / f"v3_1_{s}_significance_tables.md",
            n_resamples=10000,
            seed=42,
        )

    master_out = {
        "dataset": "SYNTH-MED-V3.1",
        "splits": list(SPLITS),
        "total_runtime_s": round(time.perf_counter() - t_start, 2),
        "significance_by_split": sig_summaries,
    }
    master_path = out_dir / "v3_1_final_locked_eval.json"
    master_path.write_text(json.dumps(master_out, indent=2), encoding="utf-8")
    print(f"Saved final 5-seed locked evaluation to {master_path} in {master_out['total_runtime_s']}s")


if __name__ == "__main__":
    main()
