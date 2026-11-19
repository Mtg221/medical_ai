"""Evaluate `deidentify_clinical_text` on the 60-sentence blind-agent EN+FR held-out dataset (written before reading redaction code)."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medresearch_platform.security.privacy import deidentify_clinical_text

_STOP_TOKENS = {
    "dr", "pr", "mr", "mrs", "ms", "mme", "mlle", "prof", "rue", "avenue", "boulevard",
    "blvd", "street", "lane", "apt", "suite", "impasse", "allée", "lot", "cité", "bât",
    "de", "la", "des", "du", "le", "les", "the", "of",
}


def _find_remaining_tokens(target: str, scrubbed: str) -> list[str]:
    """Return identifying tokens (>=3 alnum chars, non-stopword) from `target` still present verbatim in `scrubbed`."""
    scrubbed_no_tags = re.sub(r"\[[A-Z0-9_]+_REDACTED\]", " ", scrubbed)
    leaked: list[str] = []
    for tok in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]{3,}", target):
        if tok.lower() in _STOP_TOKENS:
            continue
        if re.search(rf"(?<![A-Za-zÀ-ÖØ-öø-ÿ0-9]){re.escape(tok)}(?![A-Za-zÀ-ÖØ-öø-ÿ0-9])", scrubbed_no_tags):
            leaked.append(tok)
    return leaked


def eval_blind_redaction(
    dataset_path: str | Path = "docs/results/p10_redaction_blind_agent_dataset.jsonl",
    out_json: str | Path = "docs/results/p10_redaction_blind_agent_eval.json",
    *,
    post_hoc: bool = False,
) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in Path(dataset_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    tp = fn = fp = 0
    strict_tp = strict_fn = 0
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fn": 0})
    by_lang: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0})
    missed: list[dict[str, Any]] = []
    partial_leaks: list[dict[str, Any]] = []
    false_positives: list[dict[str, Any]] = []

    for rec in records:
        text = str(rec["text"])
        lang = str(rec["language"])
        phi_items = rec.get("phi_items") or []
        for item in phi_items:
            assert item["text"] in text, f"Target {item['text']!r} not in {rec['id']}"

        res = deidentify_clinical_text(text)
        for item in phi_items:
            target = str(item["text"])
            ptype = str(item["type"])
            leaked_tokens = _find_remaining_tokens(target, res.scrubbed_text)
            if not leaked_tokens and target not in res.scrubbed_text:
                strict_tp += 1
            else:
                strict_fn += 1
                if target not in res.scrubbed_text and leaked_tokens:
                    partial_leaks.append(
                        {
                            "id": rec["id"],
                            "language": lang,
                            "type": ptype,
                            "target": target,
                            "leaked_tokens": leaked_tokens,
                            "scrubbed": res.scrubbed_text,
                        }
                    )
            # Exact match to eval_p10_redaction.py criterion: target substring absent from scrubbed_text.
            if target not in res.scrubbed_text:
                tp += 1
                by_type[ptype]["tp"] += 1
                by_lang[lang]["tp"] += 1
            else:
                fn += 1
                by_type[ptype]["fn"] += 1
                by_lang[lang]["fn"] += 1
                missed.append(
                    {
                        "id": rec["id"],
                        "language": lang,
                        "type": ptype,
                        "missed": target,
                        "text": text,
                        "scrubbed": res.scrubbed_text,
                    }
                )
        if not phi_items and res.detected_spans:
            fp += len(res.detected_spans)
            by_lang[lang]["fp"] += len(res.detected_spans)
            false_positives.append(
                {
                    "id": rec["id"],
                    "language": lang,
                    "text": text,
                    "scrubbed": res.scrubbed_text,
                    "spans": [{"type": s.entity_type, "value": s.original_value} for s in res.detected_spans],
                }
            )

    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    type_summary = {
        k: {
            "tp": v["tp"],
            "fn": v["fn"],
            "total": v["tp"] + v["fn"],
            "recall": round(v["tp"] / max(1, v["tp"] + v["fn"]), 4),
        }
        for k, v in sorted(by_type.items())
    }
    lang_summary = {
        k: {
            "tp": v["tp"],
            "fn": v["fn"],
            "fp": v["fp"],
            "recall": round(v["tp"] / max(1, v["tp"] + v["fn"]), 4),
        }
        for k, v in sorted(by_lang.items())
    }

    report: dict[str, Any] = {
        "evaluated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": str(dataset_path),
        "authored_by": "blind_agent_before_reading_redaction_code",
        "evaluation_mode": "post_hoc_seen_after_fix" if post_hoc else "blind_before_reading_redaction_code",
        "blind": not post_hoc,
        "owner_written": False,
        "n_sentences": len(records),
        "n_phi_sentences": sum(1 for r in records if r.get("has_phi")),
        "n_non_phi_sentences": sum(1 for r in records if not r.get("has_phi")),
        "n_phi_items": tp + fn,
        "tp": tp,
        "fn": fn,
        "fp_on_non_phi_sentences": fp,
        "recall": round(recall, 4),
        "strict_full_token_tp": strict_tp,
        "strict_full_token_fn": strict_fn,
        "strict_full_token_recall": round(strict_tp / max(1, strict_tp + strict_fn), 4),
        "partial_leaks_count": len(partial_leaks),
        "precision": round(precision, 4),
        "by_type": type_summary,
        "by_language": lang_summary,
        "missed": missed,
        "partial_leaks": partial_leaks,
        "false_positives": false_positives,
    }
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="docs/results/p10_redaction_blind_agent_dataset.jsonl")
    parser.add_argument("--out", default="docs/results/p10_redaction_blind_agent_eval.json")
    parser.add_argument("--post-hoc", action="store_true", help="Mark run as post-hoc (evaluated after inspecting/fixing redactor)")
    args = parser.parse_args()
    print(json.dumps(eval_blind_redaction(args.dataset, args.out, post_hoc=args.post_hoc), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
