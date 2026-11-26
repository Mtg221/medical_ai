"""Step 2 (L3/L4): Re-measure assertion macro-F1 (overall, EN, FR) after general ConText scope rules."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from medresearch_platform.approaches.classical_ml import ClassicalCRFModel
from medresearch_platform.eval_harness.metrics import Evaluator
from medresearch_platform.eval_harness.runner import APPROACH_REGISTRY, _load_jsonl
from medresearch_platform.hybrid.pipeline import HybridPipeline


def _score_langs(ev: Evaluator, docs: list[Any], preds: list[Any]) -> dict[str, float]:
    en_docs = [d for d in docs if d.language == "en"]
    fr_docs = [d for d in docs if d.language == "fr"]
    en_preds = [p for d, p in zip(docs, preds) if d.language == "en"]
    fr_preds = [p for d, p in zip(docs, preds) if d.language == "fr"]
    return {
        "overall_assertion_macro_f1": round(float(ev.evaluate(docs, preds)["assertion"]["macro_f1"]), 4),
        "en_assertion_macro_f1": round(float(ev.evaluate(en_docs, en_preds)["assertion"]["macro_f1"]), 4),
        "fr_assertion_macro_f1": round(float(ev.evaluate(fr_docs, fr_preds)["assertion"]["macro_f1"]), 4),
    }


def main() -> None:
    t0 = time.perf_counter()
    ev = Evaluator()
    pipe = HybridPipeline.load(Path("models/hybrid_v2_seed1"))
    v3_dir = Path("data/processed/v3_1")
    train_docs = _load_jsonl(v3_dir / "train.jsonl")
    crf = ClassicalCRFModel(seed=1)
    crf.fit(train_docs)

    out: dict[str, Any] = {"splits": {}}
    for sp in ("val", "test_iid", "test_unseen_terms", "test_unseen_templates"):
        docs = _load_jsonl(v3_dir / f"{sp}.jsonl")
        h2_preds = [pipe.extract_document(d)[0] for d in docs]
        crf_preds = [crf.predict_document(d) for d in docs]
        rb_preds = APPROACH_REGISTRY["rule_based"]([], docs, 1, {})
        res = {
            "hybrid_v2_seed1": _score_langs(ev, docs, h2_preds),
            "classical_crf_seed1": _score_langs(ev, docs, crf_preds),
            "rule_based": _score_langs(ev, docs, rb_preds),
        }
        out["splits"][sp] = res
        print(f"[{sp}] {res}")

    out["runtime_s"] = round(time.perf_counter() - t0, 2)
    Path("docs/results/v3_2_l3_l4_assertion_recheck.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Done in {out['runtime_s']}s")


if __name__ == "__main__":
    main()
