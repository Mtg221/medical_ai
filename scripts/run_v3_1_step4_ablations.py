"""Run 5-seed Hybrid Stage 1-5 ablation on SYNTH-MED-V3.1 (val, test_unseen_terms, test_unseen_templates).

Records:
- Multi-seed mean, std, and 95% CI for all 6 ablation variants on val, test_unseen_terms, test_unseen_templates
- Per-class assertion F1 & confusion matrix (full_hybrid vs no_assertion_bilstm) on val
- Per-type TIMEX normalization accuracy (DATE, RELATIVE, DURATION, FREQUENCY) on val
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from medresearch_platform.data.schema import ASSERTION_CLASSES, AnnotatedDocument
from medresearch_platform.eval_harness.metrics import DocumentPrediction, Evaluator
from medresearch_platform.hybrid.pipeline import HybridPipeline

SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5)
SPLITS: tuple[str, ...] = ("val", "test_unseen_terms", "test_unseen_templates")
VARIANTS: list[tuple[str, dict[str, bool]]] = [
    ("full_hybrid", {}),
    ("no_rules_layer", {"use_rules_layer": False}),
    ("no_assertion_bilstm", {"use_assertion_bilstm": False}),
    ("no_relation_cnn", {"use_relation_cnn": False}),
    ("no_temporal_crf", {"use_temporal_crf": False}),
    ("no_transformer_fusion", {"use_transformer_fusion": False}),
]
METRIC_KEYS: tuple[str, ...] = (
    "strict_f1",
    "relaxed_f1",
    "assertion_macro_f1",
    "relation_f1",
    "timex_f1",
    "timex_norm_acc",
    "doc_type_acc",
)


def _load_split(path: Path) -> list[AnnotatedDocument]:
    with path.open("r", encoding="utf-8") as fh:
        return [AnnotatedDocument.from_dict(json.loads(line)) for line in fh if line.strip()]


def _extract_metrics(raw: dict[str, Any]) -> dict[str, float]:
    return {
        "strict_f1": float(raw["strict_span"]["f1"]),
        "relaxed_f1": float(raw["lenient_span"]["f1"]),
        "assertion_macro_f1": float(raw["assertion"]["macro_f1"]),
        "relation_f1": float(raw["relations"]["f1"]),
        "timex_f1": float(raw["timex"]["f1"]),
        "timex_norm_acc": float(raw["timex"]["normalization_accuracy"]),
        "doc_type_acc": float(raw["doc_classification"]["accuracy"]),
    }


def _summarize_seeds(seed_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    n = len(seed_metrics)
    out: dict[str, dict[str, float]] = {}
    for k in METRIC_KEYS:
        vals = [m[k] for m in seed_metrics]
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        ci95 = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
        out[k] = {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "ci95_low": round(max(0.0, mean - ci95), 4),
            "ci95_high": round(min(1.0, mean + ci95), 4),
        }
    return out


def _assertion_diagnostics(
    docs: list[AnnotatedDocument], preds: list[DocumentPrediction]
) -> dict[str, Any]:
    cm: dict[str, dict[str, int]] = {g: {p: 0 for p in ASSERTION_CLASSES} for g in ASSERTION_CLASSES}
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    for d, pred in zip(docs, preds):
        pred_map = {(e.type, e.start, e.end): e.assertion for e in pred.entities}
        for ge in d.entities:
            pa = pred_map.get((ge.type, ge.start, ge.end))
            if pa is None:
                continue
            cm[ge.assertion][pa] += 1
            if pa == ge.assertion:
                tp[ge.assertion] += 1
            else:
                fn[ge.assertion] += 1
                fp[pa] += 1
    per_class: dict[str, dict[str, float]] = {}
    for cls in ASSERTION_CLASSES:
        p = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0.0
        r = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        per_class[cls] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "support": int(tp[cls] + fn[cls]),
        }
    return {"per_class": per_class, "confusion_matrix": cm}


def _timex_norm_by_type(
    docs: list[AnnotatedDocument], preds: list[DocumentPrediction]
) -> dict[str, dict[str, Any]]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [correct, total]
    for d, pred in zip(docs, preds):
        pred_map = {(t.type, t.start, t.end): t.normalized for t in pred.timex}
        for gt in d.timex:
            if not gt.normalized:
                continue
            counts[gt.type][1] += 1
            if pred_map.get((gt.type, gt.start, gt.end)) == gt.normalized:
                counts[gt.type][0] += 1
    return {
        tp: {
            "correct": c,
            "total": tot,
            "accuracy": round(c / tot, 4) if tot > 0 else 0.0,
        }
        for tp, (c, tot) in sorted(counts.items())
    }


def main() -> None:
    t_start = time.perf_counter()
    data_dir = Path("data/processed/v3_1")
    train_docs = _load_split(data_dir / "train.jsonl")
    split_docs = {s: _load_split(data_dir / f"{s}.jsonl") for s in SPLITS}

    cfg = yaml.safe_load(Path("configs/exp/v3_1/hybrid.yaml").read_text(encoding="utf-8"))
    evaluator = Evaluator()

    per_split_seed_metrics: dict[str, dict[str, list[dict[str, float]]]] = {
        s: {vname: [] for vname, _ in VARIANTS} for s in SPLITS
    }
    val_assert_preds: dict[str, list[tuple[list[AnnotatedDocument], list[DocumentPrediction]]]] = {
        "full_hybrid": [],
        "no_assertion_bilstm": [],
    }
    val_timex_preds: dict[str, list[tuple[list[AnnotatedDocument], list[DocumentPrediction]]]] = {
        "full_hybrid": [],
        "no_temporal_crf": [],
    }
    fit_times_s: list[float] = []

    for seed in SEEDS:
        t0 = time.perf_counter()
        pipe = HybridPipeline(seed=seed, cfg=cfg).fit(train_docs)
        fit_s = round(time.perf_counter() - t0, 2)
        fit_times_s.append(fit_s)
        print(f"[seed={seed}] HybridPipeline fit in {fit_s}s", flush=True)

        for split_name in SPLITS:
            docs = split_docs[split_name]
            for vname, kwargs in VARIANTS:
                preds = [pipe.extract_document(d, **kwargs)[0] for d in docs]
                m = _extract_metrics(evaluator.evaluate(docs, preds))
                per_split_seed_metrics[split_name][vname].append(m)
                if split_name == "val":
                    if vname in val_assert_preds:
                        val_assert_preds[vname].append((docs, preds))
                    if vname in val_timex_preds:
                        val_timex_preds[vname].append((docs, preds))
                if vname == "full_hybrid":
                    print(
                        f"  [seed={seed}][{split_name}][{vname}] "
                        f"strict={m['strict_f1']:.4f} assert={m['assertion_macro_f1']:.4f} "
                        f"rel={m['relation_f1']:.4f} timex={m['timex_f1']:.4f} norm={m['timex_norm_acc']:.4f}",
                        flush=True,
                    )

    summary: dict[str, Any] = {
        "dataset": "SYNTH-MED-V3.1",
        "seeds": list(SEEDS),
        "fit_times_s": fit_times_s,
        "total_runtime_s": round(time.perf_counter() - t_start, 2),
        "splits": {},
        "val_diagnostics": {},
    }

    for split_name in SPLITS:
        full_summary = _summarize_seeds(per_split_seed_metrics[split_name]["full_hybrid"])
        split_out: dict[str, Any] = {}
        for vname, _ in VARIANTS:
            vs = _summarize_seeds(per_split_seed_metrics[split_name][vname])
            deltas = {
                k: round(vs[k]["mean"] - full_summary[k]["mean"], 4) for k in METRIC_KEYS
            }
            split_out[vname] = {
                "summary": vs,
                "delta_vs_full": deltas,
                "per_seed": per_split_seed_metrics[split_name][vname],
            }
        summary["splits"][split_name] = split_out

    # Pooled diagnostics across all 5 seeds on val
    for vname, pairs in val_assert_preds.items():
        all_docs = [d for docs, _ in pairs for d in docs]
        all_preds = [p for _, preds in pairs for p in preds]
        summary["val_diagnostics"][f"assertion_{vname}"] = _assertion_diagnostics(all_docs, all_preds)

    for vname, pairs in val_timex_preds.items():
        all_docs = [d for docs, _ in pairs for d in docs]
        all_preds = [p for _, preds in pairs for p in preds]
        summary["val_diagnostics"][f"timex_norm_{vname}"] = _timex_norm_by_type(all_docs, all_preds)

    out_path = Path("docs/results/v3_1_step4_ablations.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved Step 4 ablation results to {out_path} in {summary['total_runtime_s']}s")


if __name__ == "__main__":
    main()
