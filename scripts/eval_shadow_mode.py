"""Prospective Shadow-Mode Clinical Safety & Fairness Evaluation Harness (Phase 4).

Evaluates shadow-mode hospital logs (JSONL) against prospective ground truth (biopsy /
adjudicated clinical consensus) and checks mandatory safety promotion gates across:
  - Overall and per-modality AUROC, Sensitivity @ 90% Specificity, NPV, ECE, Brier
  - Fitzpatrick skin-tone subgroup parity (I-II vs III-IV vs V-VI max sensitivity gap <= 0.08)
  - Uncertainty abstention calibration & clinician override/agreement rates
  - Serving latency p50 / p95 SLOs
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from medresearch_platform.vision.eval_vision import (
    compute_auroc,
    compute_ece_and_reliability,
    compute_sensitivity_at_specificity,
)

DEFAULT_PROMOTION_GATES: dict[str, float] = {
    "min_dermoscopic_auroc": 0.94,
    "min_dermoscopic_sens_at_90_spec": 0.90,
    "min_clinical_auroc": 0.88,
    "min_clinical_sens_at_90_spec": 0.75,
    "max_ece": 0.05,
    "max_fitzpatrick_sens_gap": 0.08,
    "max_p95_latency_ms": 2500.0,
    "min_clinician_agreement_rate": 0.85,
}


def _fitzpatrick_bucket(tone: str | None) -> str:
    raw = str(tone or "UNKNOWN").strip().upper()
    if raw in {"I", "II", "1", "2", "I-II", "LIGHT"}:
        return "I-II"
    if raw in {"III", "IV", "3", "4", "III-IV", "MEDIUM"}:
        return "III-IV"
    if raw in {"V", "VI", "5", "6", "V-VI", "DARK"}:
        return "V-VI"
    return "UNKNOWN"


def _compute_group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labeled = [r for r in rows if r.get("ground_truth_malignant") in (0, 1) and not r.get("abstained", False)]
    abstained = [r for r in rows if r.get("abstained", False)]
    n_total = len(rows)
    out: dict[str, Any] = {
        "n_total": n_total,
        "n_scored": len(labeled),
        "n_abstained": len(abstained),
        "abstention_rate": round(len(abstained) / n_total, 4) if n_total else 0.0,
    }
    if not labeled:
        out["status"] = "no_scored_cases"
        return out

    y = np.array([int(r["ground_truth_malignant"]) for r in labeled], dtype=np.int64)
    p = np.array([float(r["predicted_prob_malignant"]) for r in labeled], dtype=np.float64)
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    out["malignant"] = pos
    out["benign"] = neg
    if pos == 0 or neg == 0:
        out["status"] = "single_class"
        return out

    ece, _ = compute_ece_and_reliability(y, p)
    pred_pos = p >= 0.5
    tn = int(np.sum((~pred_pos) & (y == 0)))
    fn = int(np.sum((~pred_pos) & (y == 1)))
    npv = tn / max(1, tn + fn)

    out.update(
        {
            "auroc": round(float(compute_auroc(y, p)), 4),
            "sens_at_90_spec": round(float(compute_sensitivity_at_specificity(y, p, target_spec=0.90)), 4),
            "npv_at_050": round(float(npv), 4),
            "ece": round(float(ece), 4),
            "brier": round(float(np.mean((p - y) ** 2)), 4),
        }
    )
    return out


def evaluate_shadow_mode_records(
    records: list[dict[str, Any]],
    *,
    gates: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute shadow-mode clinical metrics, Fitzpatrick parity, latency SLOs, and promotion gates."""
    eff_gates = {**DEFAULT_PROMOTION_GATES, **(gates or {})}
    overall = _compute_group_metrics(records)

    by_modality: dict[str, Any] = {}
    for mod in sorted({str(r.get("modality", "unknown")).lower() for r in records}):
        by_modality[mod] = _compute_group_metrics(
            [r for r in records if str(r.get("modality", "unknown")).lower() == mod]
        )

    by_fitzpatrick: dict[str, Any] = {}
    for bucket in ("I-II", "III-IV", "V-VI", "UNKNOWN"):
        sub = [r for r in records if _fitzpatrick_bucket(r.get("fitzpatrick")) == bucket]
        if sub:
            by_fitzpatrick[bucket] = _compute_group_metrics(sub)

    # Fitzpatrick parity gap across known buckets with valid sens_at_90_spec
    known_sens = [
        float(by_fitzpatrick[b]["sens_at_90_spec"])
        for b in ("I-II", "III-IV", "V-VI")
        if b in by_fitzpatrick and "sens_at_90_spec" in by_fitzpatrick[b]
    ]
    fitz_gap = round(max(known_sens) - min(known_sens), 4) if len(known_sens) >= 2 else None

    latencies = [float(r["latency_ms"]) for r in records if r.get("latency_ms") is not None]
    latency_stats = {
        "n": len(latencies),
        "p50_ms": round(float(np.percentile(latencies, 50)), 2) if latencies else None,
        "p95_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else None,
    }

    reviewed = [r for r in records if r.get("clinician_agreed") is not None]
    agreement_rate = (
        round(sum(1 for r in reviewed if bool(r["clinician_agreed"])) / len(reviewed), 4)
        if reviewed
        else None
    )

    gate_checks: dict[str, dict[str, Any]] = {}

    def _check(name: str, actual: float | None, threshold: float, op: str) -> None:
        if actual is None:
            gate_checks[name] = {"status": "INSUFFICIENT_DATA", "actual": None, "threshold": threshold}
            return
        passed = (actual >= threshold) if op == ">=" else (actual <= threshold)
        gate_checks[name] = {
            "status": "PASS" if passed else "FAIL",
            "actual": actual,
            "threshold": threshold,
        }

    derm = by_modality.get("dermoscopic", {})
    clin = by_modality.get("clinical", {})
    _check("dermoscopic_auroc", derm.get("auroc"), eff_gates["min_dermoscopic_auroc"], ">=")
    _check("dermoscopic_sens_at_90_spec", derm.get("sens_at_90_spec"), eff_gates["min_dermoscopic_sens_at_90_spec"], ">=")
    _check("clinical_auroc", clin.get("auroc"), eff_gates["min_clinical_auroc"], ">=")
    _check("clinical_sens_at_90_spec", clin.get("sens_at_90_spec"), eff_gates["min_clinical_sens_at_90_spec"], ">=")
    _check("overall_ece", overall.get("ece"), eff_gates["max_ece"], "<=")
    _check("fitzpatrick_sens_gap", fitz_gap, eff_gates["max_fitzpatrick_sens_gap"], "<=")
    _check("p95_latency_ms", latency_stats["p95_ms"], eff_gates["max_p95_latency_ms"], "<=")
    _check("clinician_agreement_rate", agreement_rate, eff_gates["min_clinician_agreement_rate"], ">=")

    all_passed = all(g["status"] == "PASS" for g in gate_checks.values())
    any_failed = any(g["status"] == "FAIL" for g in gate_checks.values())
    verdict = "READY_FOR_SUPERVISED_PILOT" if all_passed else ("BLOCKED_FAILED_GATES" if any_failed else "PENDING_MORE_DATA")

    return {
        "evaluated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "verdict": verdict,
        "overall": overall,
        "by_modality": by_modality,
        "by_fitzpatrick": by_fitzpatrick,
        "fitzpatrick_sens_gap": fitz_gap,
        "latency": latency_stats,
        "clinician_agreement_rate": agreement_rate,
        "promotion_gates": gate_checks,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate prospective shadow-mode clinical logs (JSONL).")
    p.add_argument("--input", type=Path, required=True, help="Path to shadow-mode JSONL records.")
    p.add_argument("--out", type=Path, default=Path("docs/results/shadow_mode_eval.json"))
    args = p.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = evaluate_shadow_mode_records(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
