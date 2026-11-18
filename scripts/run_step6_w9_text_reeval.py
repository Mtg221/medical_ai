"""Step 6 (W9): Rebuild served hybrid_v2_seed1 and re-evaluate on val + 3 locked text test sets."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import yaml

from medresearch_platform.eval_harness.metrics import Evaluator
from medresearch_platform.eval_harness.runner import _load_jsonl
from medresearch_platform.hybrid.pipeline import HybridPipeline
from medresearch_platform.vision.splits import log_locked_test_access


def run_w9_text_reeval(
    out_json: str | Path = "docs/results/w9_text_reeval.json",
    model_out_dir: str | Path = "models/hybrid_v2_seed1",
    log_test_set: bool = True,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    cfg = yaml.safe_load(Path("configs/exp/v3_2/hybrid_v2.yaml").read_text(encoding="utf-8"))
    train_docs = _load_jsonl(Path("data/processed/v3_1/train.jsonl"))

    pipe = HybridPipeline(seed=1, cfg=cfg).fit(train_docs)
    pipe.save(Path(model_out_dir), corpus_id="SYNTH-MED-V3.1")

    evaluator = Evaluator()
    prior_path = Path("docs/results/v3_2_hybrid_v2_locked_eval.json")
    prior = json.loads(prior_path.read_text(encoding="utf-8")) if prior_path.exists() else {}

    splits_out: dict[str, Any] = {}
    locked_dir = Path("data/locked/synth_med_v3_1")
    for sp in ("val", "test_iid", "test_unseen_terms", "test_unseen_templates"):
        docs = _load_jsonl(locked_dir / f"{sp}.jsonl")
        preds = [pipe.extract_document(d)[0] for d in docs]
        ev = evaluator.evaluate(docs, preds)
        prior_sp = prior.get("splits", {}).get(sp, {}).get("hybrid_v2", {})
        splits_out[sp] = {
            "n_docs": len(docs),
            "after_w9": ev,
            "before_w9_5seed_summary": {
                k: prior_sp.get(k)
                for k in ("strict_f1_mean", "lenient_f1_mean", "assertion_macro_f1_mean", "relation_f1_mean")
                if k in prior_sp
            },
        }

    if log_test_set:
        log_locked_test_access(
            dataset_name="SYNTH-MED-V3.1:{val,test_iid,test_unseen_terms,test_unseen_templates}",
            model_name="hybrid_v2_seed1",
            sample_count=480,
            reason="W9 person / age / rules-out",
        )

    report = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "model_dir": str(model_out_dir),
        "reason": "W9 person / age / rules-out",
        "splits": splits_out,
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    res = run_w9_text_reeval()
    for sp, blk in res["splits"].items():
        ev = blk["after_w9"]
        print(
            sp,
            "strict_f1=",
            ev.get("strict_f1"),
            "assertion_macro_f1=",
            ev.get("assertion_macro_f1"),
            "by_language=",
            ev.get("by_language"),
        )
