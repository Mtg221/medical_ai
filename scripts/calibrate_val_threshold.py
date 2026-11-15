"""Step 3 (K4): Measure val calibration (ECE, reliability bins, per-type) and select review threshold."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from medresearch_platform.ai.evaluation import compute_calibration_and_selective_risk
from medresearch_platform.eval_harness.runner import _load_jsonl
from medresearch_platform.hybrid.pipeline import HybridPipeline


def main() -> None:
    pipe = HybridPipeline.load(Path("models/hybrid_v2_seed1"))
    val_docs = _load_jsonl(Path("data/processed/v3_1/val.jsonl"))

    confs: list[float] = []
    correct: list[int] = []
    by_type_confs: dict[str, list[float]] = defaultdict(list)
    by_type_correct: dict[str, list[int]] = defaultdict(list)

    for doc in val_docs:
        pred, _ = pipe.extract_document(doc)
        gold_spans = {(e.type, e.start, e.end) for e in doc.entities}
        for e in pred.entities:
            c = float(e.confidence)
            hit = 1 if (e.type, e.start, e.end) in gold_spans else 0
            confs.append(c)
            correct.append(hit)
            by_type_confs[e.type].append(c)
            by_type_correct[e.type].append(hit)

    overall = compute_calibration_and_selective_risk(confs, correct, num_bins=10)
    # Inspect fine-grained precision by 0.10 confidence bins [0.0..1.0]
    bins: list[dict[str, Any]] = []
    arr_c = np.asarray(confs)
    arr_y = np.asarray(correct)
    for lo_idx in range(10):
        lo = lo_idx * 0.1
        hi = (lo_idx + 1) * 0.1
        mask = (arr_c >= lo) & (arr_c < hi if lo_idx < 9 else arr_c <= 1.0)
        cnt = int(mask.sum())
        prec = round(float(arr_y[mask].mean()), 4) if cnt > 0 else None
        bins.append({"bin": f"[{lo:.1f}, {hi:.1f}]", "count": cnt, "precision": prec})

    # Find highest confidence bin upper bound where precision < 0.80, or default 0.50 if all >= 0.80
    below_80 = [b for b in bins if b["count"] > 0 and b["precision"] is not None and b["precision"] < 0.80]
    if below_80:
        last_bin = below_80[-1]["bin"]
        threshold = float(last_bin.split(",")[1].strip("] "))
    else:
        threshold = 0.50

    per_type: dict[str, Any] = {}
    for etype in sorted(by_type_confs):
        tc = by_type_confs[etype]
        ty = by_type_correct[etype]
        cal = compute_calibration_and_selective_risk(tc, ty, num_bins=5)
        per_type[etype] = {
            "count": len(tc),
            "precision": round(sum(ty) / max(1, len(ty)), 4),
            "mean_conf": round(sum(tc) / max(1, len(tc)), 4),
            "ece": round(float(cal["ece"]), 4),
        }

    report = {
        "split": "val",
        "total_predicted_entities": len(confs),
        "overall_ece": round(float(overall["ece"]), 4),
        "reliability_bins": bins,
        "selected_review_threshold": threshold,
        "per_entity_type": per_type,
    }
    Path("docs/results/v3_2_val_calibration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
