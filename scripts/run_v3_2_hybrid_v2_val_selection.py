"""Step 4 (12B): Select Hybrid v2 design strictly on `val` (5 seeds), then run once on locked test splits."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import yaml

from medresearch_platform.data.schema import AnnotatedDocument
from medresearch_platform.eval_harness.metrics import (
    DocumentPrediction,
    Evaluator,
    bootstrap_f1_ci,
    paired_significance_test,
)
from medresearch_platform.eval_harness.runner import APPROACH_REGISTRY, _load_jsonl

SEEDS = [1, 2, 3, 4, 5]


def _val_design_grid() -> list[tuple[str, dict[str, Any]]]:
    base = yaml.safe_load(Path("configs/exp/v3_1/hybrid.yaml").read_text(encoding="utf-8"))
    return [
        ("v3_1_baseline", {**base, "use_crf_ensemble": False, "use_assertion_bilstm": True, "use_transformer_fusion": True}),
        ("crf_off_ctx_nofusion", {**base, "use_crf_ensemble": False, "use_assertion_bilstm": False, "use_transformer_fusion": False}),
        ("crf_on_bilstm_fusion", {**base, "use_crf_ensemble": True, "use_assertion_bilstm": True, "use_transformer_fusion": True}),
        ("crf_on_ctx_fusion", {**base, "use_crf_ensemble": True, "use_assertion_bilstm": False, "use_transformer_fusion": True}),
        ("crf_on_ctx_nofusion", {**base, "use_crf_ensemble": True, "use_assertion_bilstm": False, "use_transformer_fusion": False}),
    ]


def _summarize_with_doc_bootstrap(
    docs: list[AnnotatedDocument],
    preds_by_seed: list[list[DocumentPrediction]],
) -> dict[str, Any]:
    evaluator = Evaluator()
    per_seed = [evaluator.evaluate(docs, p) for p in preds_by_seed]
    strict_vals = [float(m["strict_span"]["f1"]) for m in per_seed]
    lenient_vals = [float(m["lenient_span"]["f1"]) for m in per_seed]
    assert_vals = [float(m["assertion"]["macro_f1"]) for m in per_seed]
    rel_vals = [float(m["relations"]["f1"]) for m in per_seed]
    timex_vals = [float(m["timex"]["f1"]) for m in per_seed]
    norm_vals = [float(m["timex"]["normalization_accuracy"]) for m in per_seed]

    # Pooled document-level bootstrap CI (10,000 resamples) across all seeds
    pooled_docs = docs * len(preds_by_seed)
    pooled_preds = [p for sp in preds_by_seed for p in sp]
    doc_boot_ci = bootstrap_f1_ci(pooled_docs, pooled_preds, n_resamples=10000, seed=42)

    return {
        "strict_f1_mean": round(statistics.mean(strict_vals), 4),
        "strict_f1_seed_std": round(statistics.stdev(strict_vals), 4) if len(strict_vals) > 1 else 0.0,
        "strict_f1_doc_ci95": [round(doc_boot_ci["ci_lower"], 4), round(doc_boot_ci["ci_upper"], 4)],
        "lenient_f1_mean": round(statistics.mean(lenient_vals), 4),
        "assertion_macro_f1_mean": round(statistics.mean(assert_vals), 4),
        "relation_f1_mean": round(statistics.mean(rel_vals), 4),
        "timex_f1_mean": round(statistics.mean(timex_vals), 4),
        "timex_norm_acc_mean": round(statistics.mean(norm_vals), 4),
    }


def main() -> None:
    t_total = time.perf_counter()
    v3_dir = Path("data/processed/v3_1")
    train_docs = _load_jsonl(v3_dir / "train.jsonl")
    val_docs = _load_jsonl(v3_dir / "val.jsonl")
    hybrid_fn = APPROACH_REGISTRY["hybrid"]

    # 1. Val-only selection (5 seeds)
    val_results: dict[str, Any] = {}
    for name, cfg in _val_design_grid():
        t0 = time.perf_counter()
        seed_preds = [hybrid_fn(train_docs, val_docs, seed, cfg) for seed in SEEDS]
        elapsed = round(time.perf_counter() - t0, 2)
        stats = _summarize_with_doc_bootstrap(val_docs, seed_preds)
        stats["total_5seed_runtime_s"] = elapsed
        stats["config"] = {
            "use_crf_ensemble": cfg["use_crf_ensemble"],
            "use_assertion_bilstm": cfg["use_assertion_bilstm"],
            "use_transformer_fusion": cfg["use_transformer_fusion"],
        }
        val_results[name] = stats
        print(
            f"[val][{name}] strict={stats['strict_f1_mean']:.4f} assert={stats['assertion_macro_f1_mean']:.4f} "
            f"rel={stats['relation_f1_mean']:.4f} time={elapsed:.1f}s",
            flush=True,
        )

    # Pick winner on val by combined (strict_f1 + assertion_macro_f1), breaking ties by faster runtime
    winner_name = max(
        val_results.keys(),
        key=lambda k: (
            round(val_results[k]["strict_f1_mean"] + 0.25 * val_results[k]["assertion_macro_f1_mean"], 4),
            -val_results[k]["total_5seed_runtime_s"],
        ),
    )
    winner_cfg = dict(_val_design_grid())[winner_name]
    Path("docs/results/v3_2_hybrid_v2_val_selection.json").write_text(
        json.dumps({"seeds": SEEDS, "winner": winner_name, "candidates": val_results}, indent=2),
        encoding="utf-8",
    )

    # 2. Evaluate winner ONCE on locked test splits (5 seeds)
    splits = {
        "test_iid": _load_jsonl(v3_dir / "test_iid.jsonl"),
        "test_unseen_terms": _load_jsonl(v3_dir / "test_unseen_terms.jsonl"),
        "test_unseen_templates": _load_jsonl(v3_dir / "test_unseen_templates.jsonl"),
    }
    all_test_docs: list[AnnotatedDocument] = []
    slices: dict[str, tuple[int, int]] = {}
    off = 0
    for sp, sdocs in splits.items():
        all_test_docs.extend(sdocs)
        slices[sp] = (off, off + len(sdocs))
        off += len(sdocs)

    crf_cfg = yaml.safe_load(Path("configs/exp/v3_1/classical_crf.yaml").read_text(encoding="utf-8"))
    crf_fn = APPROACH_REGISTRY["classical_crf"]

    t_eval = time.perf_counter()
    h2_test_preds = [hybrid_fn(train_docs, all_test_docs, seed, winner_cfg) for seed in SEEDS]
    crf_test_preds = [crf_fn(train_docs, all_test_docs, seed, crf_cfg) for seed in SEEDS]
    eval_runtime = round(time.perf_counter() - t_eval, 2)

    test_summary: dict[str, Any] = {}
    for sp, (lo, hi) in slices.items():
        sdocs = splits[sp]
        sp_h2 = [p[lo:hi] for p in h2_test_preds]
        sp_crf = [p[lo:hi] for p in crf_test_preds]
        h2_stats = _summarize_with_doc_bootstrap(sdocs, sp_h2)
        crf_stats = _summarize_with_doc_bootstrap(sdocs, sp_crf)
        sig = paired_significance_test(
            sdocs * len(SEEDS),
            [p for sp_p in sp_crf for p in sp_p],
            [p for sp_p in sp_h2 for p in sp_p],
            n_resamples=10000,
            seed=42,
        )
        test_summary[sp] = {
            "hybrid_v2": h2_stats,
            "classical_crf": crf_stats,
            "paired_vs_classical_crf": sig,
        }
        print(
            f"[{sp}][hybrid_v2] strict={h2_stats['strict_f1_mean']:.4f} "
            f"ci95={h2_stats['strict_f1_doc_ci95']} assert={h2_stats['assertion_macro_f1_mean']:.4f} "
            f"rel={h2_stats['relation_f1_mean']:.4f}",
            flush=True,
        )

    Path("docs/results/v3_2_hybrid_v2_locked_eval.json").write_text(
        json.dumps(
            {
                "seeds": SEEDS,
                "selected_on": "val",
                "winner_name": winner_name,
                "eval_runtime_s": eval_runtime,
                "total_runtime_s": round(time.perf_counter() - t_total, 2),
                "splits": test_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Done in {time.perf_counter() - t_total:.2f}s")


if __name__ == "__main__":
    main()
