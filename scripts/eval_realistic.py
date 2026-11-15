"""Evaluate all approaches on the owner-authored realistic evaluation JSONL once created."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from medresearch_platform.data.schema import ASSERTION_CLASSES, ENTITY_TYPES
from medresearch_platform.eval_harness.metrics import Evaluator
from medresearch_platform.eval_harness.runner import APPROACH_REGISTRY, _load_jsonl


def run_eval_realistic(
    *,
    train_path: Path = Path("data/processed/v3_1/train.jsonl"),
    realistic_path: Path = Path("data/processed/realistic/eval.jsonl"),
    out_path: Path = Path("docs/results/realistic_eval.json"),
    seed: int = 1,
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
    for d in eval_docs:
        for e in d.entities:
            if e.type not in valid_types:
                raise ValueError(f"Unknown entity type {e.type!r} in {d.doc_id}")
            if e.assertion not in valid_assertions:
                raise ValueError(f"Unknown assertion {e.assertion!r} in {d.doc_id}")

    evaluator = Evaluator()
    summary: dict[str, Any] = {}
    for approach_name, fn in APPROACH_REGISTRY.items():
        if approach_name == "dummy":
            continue
        cfg_file = Path(f"configs/exp/v3_1/{approach_name}.yaml")
        cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) if cfg_file.exists() else {}
        preds = fn(train_docs, eval_docs, seed, cfg)
        summary[approach_name] = evaluator.evaluate(eval_docs, preds)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"num_docs": len(eval_docs), "approaches": summary}, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all approaches on owner-annotated realistic evaluation set.")
    parser.add_argument("--realistic", type=Path, default=Path("data/processed/realistic/eval.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("docs/results/realistic_eval.json"))
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    try:
        run_eval_realistic(realistic_path=args.realistic, out_path=args.out, seed=args.seed)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
