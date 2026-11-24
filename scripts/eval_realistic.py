"""Evaluate all approaches (including hybrid_v2) on an annotated realistic evaluation JSONL with freeze-manifest & provenance checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from medresearch_platform.data.schema import ASSERTION_CLASSES, DOC_TYPES, ENTITY_TYPES, RELATION_TYPES
from medresearch_platform.eval_harness.metrics import Evaluator
from medresearch_platform.eval_harness.runner import APPROACH_REGISTRY, _load_jsonl


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _current_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def verify_owner_freeze_manifest(
    realistic_path: Path,
    eval_docs: list[Any],
    manifest_path: Path | None = None,
) -> tuple[dict[str, Any], str]:
    """Verify SHA-256 lock and 80-note protocol invariants before running an owner_written blind evaluation."""
    mpath = manifest_path or Path(f"{realistic_path}.manifest.json")
    if not mpath.exists():
        raise ValueError(
            f"Freeze manifest not found at {mpath}. "
            "Import the human-written Label Studio export with scripts/import_label_studio_realistic.py "
            "(strict mode, without --allow-partial-dev) before running --owner-written evaluation."
        )
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    if manifest.get("protocol_validated") is not True:
        raise ValueError(
            f"Freeze manifest {mpath} has protocol_validated={manifest.get('protocol_validated')!r}; "
            "cannot run --owner-written blind evaluation on an unvalidated or partial dataset."
        )
    actual_sha = _sha256_file(realistic_path)
    expected_sha = str(manifest.get("output_sha256", ""))
    if actual_sha != expected_sha:
        raise ValueError(
            f"SHA-256 mismatch for {realistic_path}: actual={actual_sha[:16]} != manifest={expected_sha[:16]}. "
            "The frozen blind evaluation file was modified after import."
        )
    if len(eval_docs) != 80:
        raise ValueError(f"Owner-written realistic set must contain exactly 80 docs, found {len(eval_docs)}.")
    lang_counts = Counter(d.language for d in eval_docs)
    if lang_counts.get("en", 0) != 40 or lang_counts.get("fr", 0) != 40:
        raise ValueError(f"Owner-written realistic set must be 40 EN / 40 FR, found {dict(lang_counts)}.")
    missing_types = sorted(set(DOC_TYPES) - {d.doc_type for d in eval_docs})
    if missing_types:
        raise ValueError(f"Owner-written realistic set missing DOC_TYPES: {missing_types}.")
    if int(manifest.get("double_annotated_docs", 0)) < 16:
        raise ValueError(
            f"Owner-written realistic set requires >= 16 double-annotated docs, found {manifest.get('double_annotated_docs')}."
        )
    return manifest, _sha256_file(mpath)


def run_eval_realistic(
    *,
    train_path: Path = Path("data/processed/v3_1/train.jsonl"),
    realistic_path: Path = Path("data/processed/realistic/eval.jsonl"),
    out_path: Path = Path("docs/results/realistic_eval.json"),
    seed: int = 1,
    owner_written: bool = False,
    post_hoc: bool = False,
    iaa_json_path: Path | None = None,
    manifest_path: Path | None = None,
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

    freeze_manifest: dict[str, Any] | None = None
    manifest_sha256: str | None = None
    if owner_written:
        freeze_manifest, manifest_sha256 = verify_owner_freeze_manifest(
            realistic_path, eval_docs, manifest_path=manifest_path
        )

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
    elif freeze_manifest is not None and isinstance(freeze_manifest.get("iaa_summary"), dict):
        iaa_summary = freeze_manifest["iaa_summary"]
        double_annotated_iaa = bool((iaa_summary or {}).get("double_annotated_docs", 0) >= 16)

    payload: dict[str, Any] = {
        "evaluated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _current_git_commit(),
        "dataset": str(realistic_path),
        "dataset_sha256": _sha256_file(realistic_path),
        "manifest_sha256": manifest_sha256,
        "dataset_version": (freeze_manifest or {}).get("dataset_version"),
        "protocol_validated": bool((freeze_manifest or {}).get("protocol_validated", False)),
        "owner_written": bool(owner_written),
        "post_hoc": bool(post_hoc),
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
    parser.add_argument("--manifest", type=Path, default=None, help="SHA-256 freeze manifest from import_label_studio_realistic.py")
    parser.add_argument("--iaa-json", type=Path, default=None, help="Optional IAA report from import_label_studio_realistic.py")
    parser.add_argument("--owner-written", action="store_true", help="Set ONLY when notes were authored by the human owner")
    parser.add_argument("--post-hoc", action="store_true", help="Mark run as post_hoc=true if re-evaluating an already-inspected set")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    try:
        run_eval_realistic(
            realistic_path=args.realistic,
            out_path=args.out,
            seed=args.seed,
            owner_written=args.owner_written,
            post_hoc=args.post_hoc,
            iaa_json_path=args.iaa_json,
            manifest_path=args.manifest,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
