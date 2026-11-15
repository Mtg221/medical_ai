"""Step 2 (K3): Re-measure assertion macro-F1 (overall, EN, FR) after ConText word-boundary fix."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from medresearch_platform.eval_harness.metrics import Evaluator
from medresearch_platform.eval_harness.runner import APPROACH_REGISTRY, _load_jsonl
from medresearch_platform.hybrid.pipeline import HybridPipeline


def _eval_split(pipe: HybridPipeline, split_name: str, docs: list[Any]) -> dict[str, Any]:
    ev = Evaluator()
    en_docs = [d for d in docs if d.language == "en"]
    fr_docs = [d for d in docs if d.language == "fr"]

    h2_all = [pipe.extract_document(d)[0] for d in docs]
    h2_en = [p for d, p in zip(docs, h2_all) if d.language == "en"]
    h2_fr = [p for d, p in zip(docs, h2_all) if d.language == "fr"]

    rb_all = APPROACH_REGISTRY["rule_based"]([], docs, 1, {})
    rb_en = [p for d, p in zip(docs, rb_all) if d.language == "en"]
    rb_fr = [p for d, p in zip(docs, rb_all) if d.language == "fr"]

    return {
        "hybrid_v2_seed1": {
            "overall_assertion_macro_f1": round(float(ev.evaluate(docs, h2_all)["assertion"]["macro_f1"]), 4),
            "en_assertion_macro_f1": round(float(ev.evaluate(en_docs, h2_en)["assertion"]["macro_f1"]), 4),
            "fr_assertion_macro_f1": round(float(ev.evaluate(fr_docs, h2_fr)["assertion"]["macro_f1"]), 4),
        },
        "rule_based": {
            "overall_assertion_macro_f1": round(float(ev.evaluate(docs, rb_all)["assertion"]["macro_f1"]), 4),
            "en_assertion_macro_f1": round(float(ev.evaluate(en_docs, rb_en)["assertion"]["macro_f1"]), 4),
            "fr_assertion_macro_f1": round(float(ev.evaluate(fr_docs, rb_fr)["assertion"]["macro_f1"]), 4),
        },
    }


def main() -> None:
    t0 = time.perf_counter()
    pipe = HybridPipeline.load(Path("models/hybrid_v2_seed1"))
    v3_dir = Path("data/processed/v3_1")
    out: dict[str, Any] = {"splits": {}}
    for sp in ("val", "test_iid", "test_unseen_terms", "test_unseen_templates"):
        docs = _load_jsonl(v3_dir / f"{sp}.jsonl")
        res = _eval_split(pipe, sp, docs)
        out["splits"][sp] = res
        print(f"[{sp}] hybrid_v2={res['hybrid_v2_seed1']} rule_based={res['rule_based']}")
    out["runtime_s"] = round(time.perf_counter() - t0, 2)
    Path("docs/results/v3_2_k3_assertion_recheck.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Done in {out['runtime_s']}s")


if __name__ == "__main__":
    main()
