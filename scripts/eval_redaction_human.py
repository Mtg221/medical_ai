"""Evaluate deidentify_clinical_text with span-level P/R/F1, strict full-token recall, and per-language/per-category metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medresearch_platform.security.privacy import DetectedPIISpan, deidentify_clinical_text


def _token_parts(value: str) -> list[str]:
    return [tok for tok in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", value) if len(tok) >= 2]


def _f1(p: float, r: float) -> float:
    return round(2.0 * p * r / (p + r), 4) if (p + r) > 0 else 0.0


def _resolve_gold_spans(text: str, phi_items: list[dict[str, Any]]) -> list[tuple[int, int, str, str]]:
    """Resolve each gold PHI item to `(start, end, type, text)` using explicit offsets or non-overlapping search."""
    resolved: list[tuple[int, int, str, str]] = []
    used: list[tuple[int, int]] = []
    text_low = text.lower()
    for item in phi_items:
        ptype = str(item["type"])
        target = str(item["text"])
        if "start" in item and "end" in item and item["start"] is not None and item["end"] is not None:
            s, e = int(item["start"]), int(item["end"])
            resolved.append((s, e, ptype, target))
            used.append((s, e))
            continue
        t_low = target.lower()
        search_from = 0
        found: tuple[int, int] | None = None
        while True:
            idx = text_low.find(t_low, search_from)
            if idx == -1:
                break
            cand = (idx, idx + len(target))
            if not any(not (cand[1] <= us or cand[0] >= ue) for us, ue in used):
                found = cand
                break
            search_from = idx + 1
        if found is None:
            idx = text_low.find(t_low)
            found = (idx, idx + len(target)) if idx != -1 else (-1, -1)
        resolved.append((found[0], found[1], ptype, target))
        if found[0] >= 0:
            used.append(found)
    return resolved


def _spans_overlap(s1: int, e1: int, s2: int, e2: int) -> bool:
    return s1 >= 0 and s2 >= 0 and not (e1 <= s2 or s1 >= e2)


def _normalize_pred_category(pii_type: str) -> str:
    if pii_type in ("PATIENT_NAME", "CLINICIAN_NAME"):
        return "PERSON"
    if pii_type in ("MEDICAL_RECORD_NUMBER",):
        return "MRN"
    if pii_type in ("AGE_OVER_89", "AGE"):
        return "AGE"
    return pii_type


def evaluate_redaction_jsonl(
    dataset_path: Path,
    *,
    owner_written: bool = False,
    redact_all_ages: bool = False,
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

    span_tp = 0
    span_fn = 0
    span_fp = 0
    exact_tp = 0
    exact_fn = 0
    exact_fp = 0
    strict_tp = 0
    strict_fn = 0
    non_phi_sentences_total = 0
    fp_non_phi_sentences = 0

    by_type: dict[str, dict[str, Any]] = {}
    by_lang: dict[str, dict[str, Any]] = {}
    missed: list[dict[str, Any]] = []
    partial_leaks: list[dict[str, Any]] = []
    false_positive_spans: list[dict[str, Any]] = []
    false_positives: list[dict[str, Any]] = []

    def _lang_bucket(lang: str) -> dict[str, Any]:
        return by_lang.setdefault(
            lang,
            {
                "tp": 0,
                "fn": 0,
                "fp": 0,
                "exact_tp": 0,
                "exact_fn": 0,
                "exact_fp": 0,
                "strict_tp": 0,
                "strict_fn": 0,
                "non_phi_sentences": 0,
                "fp_non_phi_sentences": 0,
            },
        )

    def _type_bucket(ptype: str) -> dict[str, Any]:
        return by_type.setdefault(
            ptype,
            {"tp": 0, "fn": 0, "fp": 0, "exact_tp": 0, "strict_tp": 0, "total": 0},
        )

    for row in rows:
        sid = str(row["id"])
        lang = str(row.get("language", "en"))
        text = str(row["text"])
        phi_items: list[dict[str, Any]] = list(row.get("phi_items", []))

        res = deidentify_clinical_text(text, redact_all_ages=redact_all_ages)
        scrubbed = res.scrubbed_text
        pred_spans: list[DetectedPIISpan] = list(res.detected_spans)
        scrubbed_no_placeholders = re.sub(r"\[[A-Z0-9_]+_REDACTED\]", " ", scrubbed)

        lb = _lang_bucket(lang)
        gold_spans = _resolve_gold_spans(text, phi_items)

        if not gold_spans:
            non_phi_sentences_total += 1
            lb["non_phi_sentences"] += 1
            if pred_spans or scrubbed != text:
                fp_non_phi_sentences += 1
                lb["fp_non_phi_sentences"] += 1
                false_positives.append({"id": sid, "language": lang, "text": text, "scrubbed": scrubbed})
            for p in pred_spans:
                span_fp += 1
                exact_fp += 1
                lb["fp"] += 1
                lb["exact_fp"] += 1
                tb = _type_bucket(_normalize_pred_category(p.pii_type))
                tb["fp"] += 1
                false_positive_spans.append(
                    {
                        "id": sid,
                        "language": lang,
                        "predicted_type": p.pii_type,
                        "text": p.original_value,
                        "start": p.start,
                        "end": p.end,
                        "in_non_phi_sentence": True,
                    }
                )
            continue

        # 1. Gold-side recall (overlap, exact-span, and strict full-token recall)
        matched_pred_indices: set[int] = set()
        exact_matched_pred_indices: set[int] = set()

        for gs, ge, ptype, target in gold_spans:
            tb = _type_bucket(ptype)
            tb["total"] += 1

            overlapping_preds = [
                idx for idx, p in enumerate(pred_spans) if _spans_overlap(gs, ge, p.start, p.end)
            ]
            exact_preds = [
                idx for idx, p in enumerate(pred_spans) if gs >= 0 and p.start == gs and p.end == ge
            ]
            for idx in overlapping_preds:
                matched_pred_indices.add(idx)
            for idx in exact_preds:
                exact_matched_pred_indices.add(idx)

            is_redacted = bool(overlapping_preds) and (target.lower() not in scrubbed.lower())
            if is_redacted:
                span_tp += 1
                tb["tp"] += 1
                lb["tp"] += 1
                if exact_preds:
                    exact_tp += 1
                    tb["exact_tp"] += 1
                    lb["exact_tp"] += 1
                else:
                    exact_fn += 1
                    lb["exact_fn"] += 1

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
                    lb["strict_fn"] += 1
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
                    tb["strict_tp"] += 1
                    lb["strict_tp"] += 1
            else:
                span_fn += 1
                exact_fn += 1
                strict_fn += 1
                tb["fn"] += 1
                lb["fn"] += 1
                lb["exact_fn"] += 1
                lb["strict_fn"] += 1
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

        # 2. Pred-side span-level False Positives (including intra-sentence FPs!)
        for idx, p in enumerate(pred_spans):
            if idx not in matched_pred_indices:
                span_fp += 1
                lb["fp"] += 1
                tb = _type_bucket(_normalize_pred_category(p.pii_type))
                tb["fp"] += 1
                false_positive_spans.append(
                    {
                        "id": sid,
                        "language": lang,
                        "predicted_type": p.pii_type,
                        "text": p.original_value,
                        "start": p.start,
                        "end": p.end,
                        "in_non_phi_sentence": False,
                    }
                )
            if idx not in exact_matched_pred_indices:
                exact_fp += 1
                lb["exact_fp"] += 1

    # Finalize per-type metrics
    for b in by_type.values():
        p_val = b["tp"] / max(1, b["tp"] + b["fp"]) if (b["tp"] + b["fp"]) > 0 else 0.0
        r_val = b["tp"] / max(1, b["total"]) if b["total"] > 0 else 0.0
        b["precision"] = round(p_val, 4)
        b["recall"] = round(r_val, 4)
        b["f1"] = _f1(p_val, r_val)
        b["strict_full_token_recall"] = round(b["strict_tp"] / max(1, b["total"]), 4) if b["total"] > 0 else 0.0

    # Finalize per-language metrics
    for lb in by_lang.values():
        p_val = lb["tp"] / max(1, lb["tp"] + lb["fp"]) if (lb["tp"] + lb["fp"]) > 0 else 0.0
        r_val = lb["tp"] / max(1, lb["tp"] + lb["fn"]) if (lb["tp"] + lb["fn"]) > 0 else 0.0
        ep_val = lb["exact_tp"] / max(1, lb["exact_tp"] + lb["exact_fp"]) if (lb["exact_tp"] + lb["exact_fp"]) > 0 else 0.0
        er_val = lb["exact_tp"] / max(1, lb["exact_tp"] + lb["exact_fn"]) if (lb["exact_tp"] + lb["exact_fn"]) > 0 else 0.0
        lb["precision"] = round(p_val, 4)
        lb["recall"] = round(r_val, 4)
        lb["f1"] = _f1(p_val, r_val)
        lb["exact_span_precision"] = round(ep_val, 4)
        lb["exact_span_recall"] = round(er_val, 4)
        lb["exact_span_f1"] = _f1(ep_val, er_val)
        lb["strict_full_token_recall"] = round(lb["strict_tp"] / max(1, lb["strict_tp"] + lb["strict_fn"]), 4)
        lb["non_phi_sentence_fp_rate"] = (
            round(lb["fp_non_phi_sentences"] / lb["non_phi_sentences"], 4) if lb["non_phi_sentences"] > 0 else 0.0
        )

    total_phi = span_tp + span_fn
    span_p = span_tp / max(1, span_tp + span_fp) if (span_tp + span_fp) > 0 else 0.0
    span_r = span_tp / max(1, total_phi) if total_phi > 0 else 0.0
    exact_p = exact_tp / max(1, exact_tp + exact_fp) if (exact_tp + exact_fp) > 0 else 0.0
    exact_r = exact_tp / max(1, exact_tp + exact_fn) if (exact_tp + exact_fn) > 0 else 0.0

    return {
        "evaluated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": str(dataset_path),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "owner_written": bool(owner_written),
        "metric_methodology": "span_level_exact_and_overlap_separated_from_sentence_fp_rate",
        "n_sentences": len(rows),
        "n_non_phi_sentences": non_phi_sentences_total,
        "n_phi_items": total_phi,
        "tp": span_tp,
        "fn": span_fn,
        "span_fp": span_fp,
        "fp_on_non_phi_sentences": fp_non_phi_sentences,
        "non_phi_sentence_fp_rate": (
            round(fp_non_phi_sentences / non_phi_sentences_total, 4) if non_phi_sentences_total > 0 else 0.0
        ),
        "precision": round(span_p, 4),
        "recall": round(span_r, 4),
        "f1": _f1(span_p, span_r),
        "exact_span_precision": round(exact_p, 4),
        "exact_span_recall": round(exact_r, 4),
        "exact_span_f1": _f1(exact_p, exact_r),
        "strict_full_token_recall": round(strict_tp / max(1, total_phi), 4),
        "by_type": dict(sorted(by_type.items())),
        "by_language": dict(sorted(by_lang.items())),
        "missed": missed,
        "partial_leaks": partial_leaks,
        "false_positive_spans": false_positive_spans,
        "false_positives": false_positives,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate bilingual PHI redaction on an unseen owner JSONL.")
    parser.add_argument("--input", type=Path, required=True, help="Path to unseen PHI redaction JSONL")
    parser.add_argument("--out", type=Path, default=Path("docs/results/p10_redaction_owner_eval.json"))
    parser.add_argument("--owner-written", action="store_true", help="Set ONLY for human owner-written sentences")
    parser.add_argument("--redact-all-ages", action="store_true", help="Also redact ages <= 89 (default: HIPAA > 89 only)")
    args = parser.parse_args()
    try:
        report = evaluate_redaction_jsonl(
            args.input,
            owner_written=args.owner_written,
            redact_all_ages=args.redact_all_ages,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Redaction eval ({report['n_sentences']} sentences, {report['n_phi_items']} PHI): "
        f"span P={report['precision']:.4f} R={report['recall']:.4f} F1={report['f1']:.4f} "
        f"(exact F1={report['exact_span_f1']:.4f}, strict_token_R={report['strict_full_token_recall']:.4f}, "
        f"non_phi_sent_FP_rate={report['non_phi_sentence_fp_rate']:.4f}) -> {args.out}"
    )


if __name__ == "__main__":
    main()
