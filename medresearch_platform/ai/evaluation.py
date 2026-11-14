"""Statistical benchmarking, calibration (ECE, Brier), selective prediction, fairness, and federated simulation."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_span_f1_metrics(
    predicted_spans: list[tuple[str, int, int, str]],
    gold_spans: list[tuple[str, int, int, str]],
) -> dict[str, float]:
    """Compute Exact and Partial Span Precision, Recall, and F1 over (doc_id, start, end, entity_type)."""
    pred_set = set(predicted_spans)
    gold_set = set(gold_spans)

    exact_tp = len(pred_set & gold_set)
    exact_prec = exact_tp / max(1, len(pred_set)) if pred_set else 0.0
    exact_rec = exact_tp / max(1, len(gold_set)) if gold_set else 0.0
    exact_f1 = (
        (2.0 * exact_prec * exact_rec) / (exact_prec + exact_rec)
        if (exact_prec + exact_rec) > 0
        else 0.0
    )

    # Partial span overlap within same document and entity_type
    partial_tp = 0
    for p_doc, p_s, p_e, p_type in predicted_spans:
        if any(
            g_doc == p_doc and g_type == p_type and max(p_s, g_s) < min(p_e, g_e)
            for g_doc, g_s, g_e, g_type in gold_spans
        ):
            partial_tp += 1

    part_prec = partial_tp / max(1, len(predicted_spans)) if predicted_spans else 0.0
    part_rec = min(1.0, partial_tp / max(1, len(gold_spans))) if gold_spans else 0.0
    part_f1 = (
        (2.0 * part_prec * part_rec) / (part_prec + part_rec)
        if (part_prec + part_rec) > 0
        else 0.0
    )

    return {
        "exact_precision": round(exact_prec, 4),
        "exact_recall": round(exact_rec, 4),
        "exact_f1": round(exact_f1, 4),
        "partial_precision": round(part_prec, 4),
        "partial_recall": round(part_rec, 4),
        "partial_f1": round(part_f1, 4),
        "predicted_count": len(predicted_spans),
        "gold_count": len(gold_spans),
    }


def compute_calibration_and_selective_risk(
    confidences: list[float],
    correctness: list[int],
    num_bins: int = 5,
) -> dict[str, Any]:
    """Compute Expected Calibration Error (ECE), Brier Score, reliability bins, and coverage-risk curve."""
    if not confidences:
        return {
            "ece": 0.0,
            "brier_score": 0.0,
            "reliability_bins": [],
            "coverage_risk_curve": [],
        }

    conf_arr = np.clip(np.asarray(confidences, dtype=np.float64), 0.0, 1.0)
    corr_arr = np.asarray(correctness, dtype=np.float64)
    n = len(conf_arr)

    brier_score = float(np.mean((conf_arr - corr_arr) ** 2))
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    bins_report: list[dict[str, Any]] = []

    for b_idx in range(num_bins):
        lo, hi = bin_edges[b_idx], bin_edges[b_idx + 1]
        if b_idx == num_bins - 1:
            mask = (conf_arr >= lo) & (conf_arr <= hi)
        else:
            mask = (conf_arr >= lo) & (conf_arr < hi)
        count = int(np.sum(mask))
        if count > 0:
            mean_conf = float(np.mean(conf_arr[mask]))
            mean_acc = float(np.mean(corr_arr[mask]))
            gap = abs(mean_acc - mean_conf)
            ece += (count / n) * gap
            bins_report.append({
                "bin_range": f"[{lo:.2f}, {hi:.2f}]",
                "count": count,
                "mean_confidence": round(mean_conf, 4),
                "empirical_accuracy": round(mean_acc, 4),
                "calibration_gap": round(gap, 4),
            })

    # Coverage-risk curve (selective prediction sorted by descending confidence)
    order = np.argsort(-conf_arr)
    sorted_corr = corr_arr[order]
    coverage_curve: list[dict[str, float]] = []
    for cov in (1.0, 0.9, 0.8, 0.7):
        k = max(1, int(round(cov * n)))
        acc_at_cov = float(np.mean(sorted_corr[:k]))
        coverage_curve.append({
            "coverage": cov,
            "retained_samples": k,
            "accuracy_at_coverage": round(acc_at_cov, 4),
            "selective_risk_error": round(1.0 - acc_at_cov, 4),
        })

    return {
        "ece": round(float(ece), 4),
        "brier_score": round(brier_score, 4),
        "reliability_bins": bins_report,
        "coverage_risk_curve": coverage_curve,
    }


def bootstrap_confidence_interval(
    values: list[float],
    *,
    n_resamples: int = 200,
    seed: int = 20261112,
) -> dict[str, float]:
    """Compute non-parametric bootstrap mean and 95% confidence interval (Section 20)."""
    if not values:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means[i] = np.mean(sample)
    return {
        "mean": round(float(np.mean(arr)), 4),
        "ci95_low": round(float(np.percentile(means, 2.5)), 4),
        "ci95_high": round(float(np.percentile(means, 97.5)), 4),
    }


def evaluate_subgroup_fairness(
    records_by_subgroup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate representation, F1, calibration, and abstention disparities across language/specialty subgroups."""
    subgroup_metrics: dict[str, Any] = {}
    f1_scores: list[float] = []

    for group_name, data in sorted(records_by_subgroup.items()):
        span_res = compute_span_f1_metrics(data.get("pred_spans", []), data.get("gold_spans", []))
        cal_res = compute_calibration_and_selective_risk(
            data.get("confidences", []),
            data.get("correctness", []),
        )
        abstained_count = int(data.get("abstained_count", 0))
        total_preds = max(1, len(data.get("pred_spans", [])))
        abstention_rate = round(abstained_count / total_preds, 4)

        f1_scores.append(span_res["exact_f1"])
        subgroup_metrics[group_name] = {
            "sample_count": int(data.get("document_count", 0)),
            "exact_f1": span_res["exact_f1"],
            "partial_f1": span_res["partial_f1"],
            "ece": cal_res["ece"],
            "brier_score": cal_res["brier_score"],
            "abstention_rate": abstention_rate,
        }

    max_f1_gap = round(max(f1_scores) - min(f1_scores), 4) if len(f1_scores) >= 2 else 0.0
    return {
        "subgroups": subgroup_metrics,
        "max_exact_f1_disparity": max_f1_gap,
        "fairness_methodology_note": (
            "Measured across language subgroups (en, fr, mixed_en_fr). "
            "No single metric proves fairness; unsupported clinical specialties remain out-of-scope."
        ),
    }


def simulate_federated_aggregation_round(
    local_site_weights: dict[str, dict[str, np.ndarray]],
    site_sample_counts: dict[str, int],
    *,
    algorithm: str = "FEDAVG",
    proximal_mu: float = 0.01,
    dp_noise_sigma: float = 0.001,
    seed: int = 20261112,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Run sample-weighted FedAvg / FedProx aggregation with poisoning outlier inspection and DP noise accounting."""
    total_samples = sum(site_sample_counts.values())
    if total_samples <= 0 or not local_site_weights:
        raise ValueError("Cannot aggregate empty federated site updates.")

    rng = np.random.default_rng(seed)
    site_names = sorted(local_site_weights.keys())
    first_site = local_site_weights[site_names[0]]

    # Poisoning / Byzantine norm defense: compute L2 norm of each site's parameter vector
    site_norms: dict[str, float] = {}
    for site in site_names:
        sq_sum = sum(float(np.sum(arr ** 2)) for arr in local_site_weights[site].values())
        site_norms[site] = round(float(np.sqrt(sq_sum)), 4)

    median_norm = float(np.median(list(site_norms.values())))
    quarantined_sites = [s for s, nrm in site_norms.items() if nrm > 3.5 * max(1.0, median_norm)]
    eligible_sites = [s for s in site_names if s not in quarantined_sites] or site_names
    eligible_samples = sum(site_sample_counts[s] for s in eligible_sites)

    aggregated: dict[str, np.ndarray] = {}
    for param_name, ref_tensor in first_site.items():
        acc = np.zeros_like(ref_tensor, dtype=np.float32)
        for site in eligible_sites:
            weight_factor = float(site_sample_counts[site]) / float(max(1, eligible_samples))
            site_tensor = local_site_weights[site][param_name]
            if algorithm == "FEDPROX":
                site_tensor = (1.0 - proximal_mu) * site_tensor + proximal_mu * ref_tensor
            acc += weight_factor * site_tensor

        if dp_noise_sigma > 0.0:
            acc += rng.normal(0.0, dp_noise_sigma, size=acc.shape).astype(np.float32)
        aggregated[param_name] = acc

    dp_epsilon_estimate = round(float(1.0 / max(0.05, dp_noise_sigma * 200.0)), 4) if dp_noise_sigma > 0 else 0.0
    report = {
        "algorithm": algorithm,
        "participating_sites": eligible_sites,
        "quarantined_sites": quarantined_sites,
        "site_l2_norms": site_norms,
        "total_eligible_samples": eligible_samples,
        "dp_noise_sigma": dp_noise_sigma,
        "dp_epsilon_spent": dp_epsilon_estimate,
    }
    return aggregated, report
