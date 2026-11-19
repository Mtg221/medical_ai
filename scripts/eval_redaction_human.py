"""Evaluate deidentify_clinical_text on an owner-written unseen EN/FR PHI redaction JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medresearch_platform.security.privacy import deidentify_clinical_text


def _token_parts(value: str) -> list[str]:
    return [tok for tok in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", value) if len(tok) >= 2]


def evaluate_redaction_jsonl(
    dataset_path: Path,
    *,
    owner_written: bool = False,
) -> dict[str, Any]:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"PHI redaction dataset not found at {dataset_path}. "
            "See docs/PHI_REDACTION_OWNER_TEMPLATE.jsonl for the expected JSONL schema."
        )
    rows = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tp = 0
    fn = 0
    strict_tp = 0
    strict_fn = 0
    fp_non_phi = 0
    by_type: dict[str, dict[str, Any]] = {}
    by_lang: dict[str, dict[str, int]] = {}
    missed: list[dict[str, Any]] = []
    partial_leaks: list[dict[str, Any]] = []
    false_positives: list[dict[str, Any]] = []

    for row in rows:
        sid = str(row["id"])
        lang = str(row.get("language", "en"))
        text = str(row["text"])
        phi_items: list[dict[str, str]] = list(row.get("phi_items", []))
        scrubbed = deidentify_clinical_text(text).scrubbed_text
        scrubbed_no_placeholders = re.sub(r"\[[A-Z0-9_]+_REDACTED\]", " ", scrubbed)

        lang_bucket = by_lang.setdefault(lang, {"tp": 0, "fn": 0, "fp": 0})
        if not phi_items:
            if scrubbed != text:
                fp_non_phi += 1
                lang_bucket["fp"] += 1
                false_positives.append({"id": sid, "language": lang, "text": text, "scrubbed": scrubbed})
            continue

        for item in phi_items:
            ptype = str(item["type"])
            target = str(item["text"])
            t_bucket = by_type.setdefault(ptype, {"tp": 0, "fn": 0, "total": 0})
            t_bucket["total"] += 1

            if target.lower() not in scrubbed.lower():
                tp += 1
                t_bucket["tp"] += 1
                lang_bucket["tp"] += 1
                leaked = [
                    tok
                    for tok in _token_parts(target)
                    if re.search(
                        rf"(?<![A-Za-zÀ-ÖØ-öø-ÿ0-9]){re.escape(tok)}(?![A-Za-zÀ-ÖØ-öø-ÿ0-9])",
                        scrubbed_no_placeholders,
                        re.IGNORECASE,
                    )
                ]
                if leaked:
                    strict_fn += 1
                    partial_leaks.append(
                        {
                            "id": sid,
                            "language": lang,
                            "type": ptype,
                            "target": target,
                            "leaked_tokens": leaked,
                            "scrubbed": scrubbed,
                        }
                    )
                else:
                    strict_tp += 1
            else:
                fn += 1
                strict_fn += 1
                t_bucket["fn"] += 1
                lang_bucket["fn"] += 1
                missed.append(
                    {
                        "id": sid,
                        "language": lang,
                        "type": ptype,
                        "missed": target,
                        "text": text,
                        "scrubbed": scrubbed,
                    }
                )

    for b in by_type.values():
        b["recall"] = round(b["tp"] / max(1, b["total"]), 4)
    for lb in by_lang.values():
        lb["recall"] = round(lb["tp"] / max(1, lb["tp"] + lb["fn"]), 4)

    total_phi = tp + fn
    return {
        "evaluated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": str(dataset_path),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "owner_written": bool(owner_written),
        "n_sentences": len(rows),
        "n_phi_items": total_phi,
        "tp": tp,
        "fn": fn,
        "fp_on_non_phi_sentences": fp_non_phi,
        "recall": round(tp / max(1, total_phi), 4),
        "strict_full_token_recall": round(strict_tp / max(1, total_phi), 4),
        "precision": round(tp / max(1, tp + fp_non_phi), 4),
        "by_type": dict(sorted(by_type.items())),
        "by_language": dict(sorted(by_lang.items())),
        "missed": missed,
        "partial_leaks": partial_leaks,
        "false_positives": false_positives,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate bilingual PHI redaction on an unseen owner JSONL.")
    parser.add_argument("--input", type=Path, required=True, help="Path to unseen PHI redaction JSONL")
    parser.add_argument("--out", type=Path, default=Path("docs/results/p10_redaction_owner_eval.json"))
    parser.add_argument("--owner-written", action="store_true", help="Set ONLY for human owner-written sentences")
    args = parser.parse_args()
    try:
        report = evaluate_redaction_jsonl(args.input, owner_written=args.owner_written)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Redaction eval ({report['n_sentences']} sentences, {report['n_phi_items']} PHI): "
        f"recall={report['recall']:.4f}, strict_token_recall={report['strict_full_token_recall']:.4f}, "
        f"precision={report['precision']:.4f} -> {args.out}"
    )


if __name__ == "__main__":
    main()
