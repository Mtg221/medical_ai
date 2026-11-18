"""Prompt 18 — Step 6: Single locked-test evaluation of retrained hybrid_v2_seed1 after val decision."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from medresearch_platform.eval_harness.metrics import Evaluator
from medresearch_platform.eval_harness.runner import _load_jsonl
from medresearch_platform.hybrid.pipeline import HybridPipeline
from medresearch_platform.vision.splits import log_locked_test_access


def main() -> None:
    t0 = time.perf_counter()
    pipe = HybridPipeline.load("models/hybrid_v2_seed1")
    evaluator = Evaluator()
    splits_out: dict[str, Any] = {}
    for sp in ("val", "test_iid", "test_unseen_terms", "test_unseen_templates"):
        docs = _load_jsonl(Path(f"data/processed/v3_1/{sp}.jsonl"))
        preds = [pipe.extract_document(d)[0] for d in docs]
        splits_out[sp] = {"n_docs": len(docs), "metrics": evaluator.evaluate(docs, preds)}

    log_locked_test_access(
        dataset_name="SYNTH-MED-V3.1:{val,test_iid,test_unseen_terms,test_unseen_templates}",
        model_name="hybrid_v2_seed1",
        sample_count=480,
        reason="P18 Step 6 X8 data + passive excluded-by-<test> ConText fix",
    )
    out = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "model_dir": "models/hybrid_v2_seed1",
        "splits": splits_out,
    }
    out_path = Path("docs/results/p18_step6_text_eval.json")
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for sp in ("val", "test_iid", "test_unseen_terms", "test_unseen_templates"):
        m = splits_out[sp]["metrics"]
        print(sp, "strict_f1:", round(m["strict_span"]["f1"], 4), "assertion_macro_f1:", round(m["assertion"]["macro_f1"], 4))


if __name__ == "__main__":
    main()
