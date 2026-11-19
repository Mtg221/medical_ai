"""Evaluate all approaches (including hybrid_v2) on an annotated realistic evaluation JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from medresearch_platform.data.schema import ASSERTION_CLASSES, ENTITY_TYPES, RELATION_TYPES
from medresearch_platform.eval_harness.metrics import Evaluator
from medresearch_platform.eval_harness.runner import APPROACH_REGISTRY, _load_jsonl


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_eval_realistic(
    *,
    train_path: Path = Path("data/processed/v3_1/train.jsonl"),
    realistic_path: Path = Path("data/processed/realistic/eval.jsonl"),
    out_path: Path = Path("docs/results/realistic_eval.json"),
    seed: int = 1,
    owner_written: bool = False,
    iaa_json_path: Path | None = None,
) -> dict[str, Any]:
    if not realistic_path.exists():
        raise FileNotFoundError(
            f"Owner-annotated realistic set not found at {realistic_path}. "
            "See docs/REALISTIC_SET_OWNER_CHECKLIST.md and scripts/import_label_studio_realistic.py."
        )
    train_docs = _load_jsonl(train_path)
    eval_docs = _load_jsonl(realistic_path)
    valid_types = set(ENTITY_TYPES)
    valid_assertions = set(ASSERTION_CLASSES)
    valid_relations = set(RELATION_TYPES)
    for d in eval_docs:
        for e in d.entities:
            if e.type not in valid_types:
                raise ValueError(f"Unknown entity type {e.type!r} in {d.doc_id}")
            if e.assertion not in valid_assertions:
                raise ValueError(f"Unknown assertion {e.assertion!r} in {d.doc_id}")
        for r in d.relations:
            if r.type not in valid_relations:
                raise ValueError(f"Unknown relation type {r.type!r} in {d.doc_id}")

    evaluator = Evaluator()
    summary: dict[str, Any] = {}
    for approach_name, fn in APPROACH_REGISTRY.items():
        if approach_name == "dummy":
            continue
        cfg_file = Path(f"configs/exp/v3_1/{approach_name}.yaml")
        cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) if cfg_file.exists() else {}
        preds = fn(train_docs, eval_docs, seed, cfg)
        summary[approach_name] = evaluator.evaluate(eval_docs, preds)

    hybrid_v2_cfg_path = Path("configs/exp/v3_2/hybrid_v2.yaml")
    if hybrid_v2_cfg_path.exists() and "hybrid" in APPROACH_REGISTRY:
        cfg_v2 = yaml.safe_load(hybrid_v2_cfg_path.read_text(encoding="utf-8"))
        preds_v2 = APPROACH_REGISTRY["hybrid"](train_docs, eval_docs, seed, cfg_v2)
        summary["hybrid_v2"] = evaluator.evaluate(eval_docs, preds_v2)

    iaa_summary: dict[str, Any] | None = None
    double_annotated_iaa = False
    if iaa_json_path is not None and iaa_json_path.exists():
        iaa_summary = json.loads(iaa_json_path.read_text(encoding="utf-8"))
        double_annotated_iaa = bool((iaa_summary or {}).get("double_annotated_docs", 0) > 0)

    payload: dict[str, Any] = {
        "evaluated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": str(realistic_path),
        "dataset_sha256": _sha256_file(realistic_path),
        "owner_written": bool(owner_written),
        "double_annotated_iaa": double_annotated_iaa,
        "iaa_summary": iaa_summary,
        "seed": seed,
        "num_docs": len(eval_docs),
        "gold_counts": {
            "entities": sum(len(d.entities) for d in eval_docs),
            "relations": sum(len(d.relations) for d in eval_docs),
            "timex": sum(len(d.timex) for d in eval_docs),
        },
        "approaches": summary,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all approaches on owner-annotated realistic evaluation set.")
    parser.add_argument("--realistic", type=Path, default=Path("data/processed/realistic/eval.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("docs/results/realistic_eval.json"))
    parser.add_argument("--iaa-json", type=Path, default=None, help="Optional IAA report from import_label_studio_realistic.py")
    parser.add_argument("--owner-written", action="store_true", help="Set ONLY when notes were authored by the human owner")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    try:
        run_eval_realistic(
            realistic_path=args.realistic,
            out_path=args.out,
            seed=args.seed,
            owner_written=args.owner_written,
            iaa_json_path=args.iaa_json,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
