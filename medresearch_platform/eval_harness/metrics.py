"""Unified evaluation metrics, calibration, bootstrap CIs, and paired significance tests."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from medresearch_platform.data.schema import ASSERTION_CLASSES, AnnotatedDocument


@dataclass(frozen=True)
class PredictedEntity:
    type: str
    start: int
    end: int
    text: str
    assertion: str = "present"
    confidence: float = 1.0
    needs_review: bool = False


@dataclass(frozen=True)
class PredictedRelation:
    type: str
    head_start: int
    head_end: int
    tail_start: int
    tail_end: int
    confidence: float = 1.0


@dataclass(frozen=True)
class PredictedTimex:
    type: str
    start: int
    end: int
    text: str
    normalized: str | None = None


@dataclass
class DocumentPrediction:
    doc_id: str
    language: str
    doc_type: str
    predicted_doc_type: str | None = None
    entities: list[PredictedEntity] = field(default_factory=list)
    relations: list[PredictedRelation] = field(default_factory=list)
    timex: list[PredictedTimex] = field(default_factory=list)
    ocr_induced: bool = False


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {"precision": round(p, 6), "recall": round(r, 6), "f1": round(f1, 6), "tp": float(tp), "fp": float(fp), "fn": float(fn)}


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def _macro_f1_from_pairs(pairs: list[tuple[str, str]], labels: tuple[str, ...] | list[str]) -> tuple[float, float]:
    if not pairs:
        return 0.0, 0.0
    correct = sum(1 for g, p in pairs if g == p)
    acc = correct / len(pairs)
    f1s: list[float] = []
    for lbl in labels:
        tp = sum(1 for g, p in pairs if g == lbl and p == lbl)
        fp = sum(1 for g, p in pairs if g != lbl and p == lbl)
        fn = sum(1 for g, p in pairs if g == lbl and p != lbl)
        if tp + fp + fn > 0:
            f1s.append(_prf(tp, fp, fn)["f1"])
    macro = sum(f1s) / len(f1s) if f1s else 0.0
    return round(acc, 6), round(macro, 6)


def compute_ece_and_brier(confidences: list[float], correct_flags: list[int], n_bins: int = 10) -> dict[str, float]:
    if not confidences:
        return {"ece": 0.0, "brier": 0.0}
    n = len(confidences)
    brier = sum((c - y) ** 2 for c, y in zip(confidences, correct_flags, strict=True)) / n
    ece = 0.0
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        idxs = [
            i
            for i, c in enumerate(confidences)
            if (lo <= c < hi) or (b == n_bins - 1 and c == 1.0)
        ]
        if not idxs:
            continue
        avg_conf = sum(confidences[i] for i in idxs) / len(idxs)
        avg_acc = sum(correct_flags[i] for i in idxs) / len(idxs)
        ece += (len(idxs) / n) * abs(avg_acc - avg_conf)
    return {"ece": round(ece, 6), "brier": round(brier, 6)}


class Evaluator:
    """Single shared evaluator for all 6 extraction approaches."""

    def evaluate(
        self,
        gold_docs: list[AnnotatedDocument],
        predictions: list[DocumentPrediction],
    ) -> dict[str, Any]:
        pred_by_id = {p.doc_id: p for p in predictions}

        strict_tp = strict_fp = strict_fn = 0
        lenient_tp = lenient_fp = lenient_fn = 0
        type_counts: dict[str, Counter[str]] = defaultdict(Counter)
        lang_counts: dict[str, Counter[str]] = defaultdict(Counter)
        doctype_counts: dict[str, Counter[str]] = defaultdict(Counter)

        assertion_pairs: list[tuple[str, str]] = []
        rel_tp = rel_fp = rel_fn = 0
        timex_tp = timex_fp = timex_fn = 0
        timex_norm_correct = timex_norm_total = 0
        doc_cls_pairs: list[tuple[str, str]] = []
        all_doc_labels: set[str] = set()

        confs: list[float] = []
        corr_flags: list[int] = []

        for gold in gold_docs:
            pred = pred_by_id.get(
                gold.doc_id,
                DocumentPrediction(doc_id=gold.doc_id, language=gold.language, doc_type=gold.doc_type),
            )
            all_doc_labels.add(gold.doc_type)
            if pred.predicted_doc_type is not None:
                all_doc_labels.add(pred.predicted_doc_type)
                doc_cls_pairs.append((gold.doc_type, pred.predicted_doc_type))

            # Strict span matching: (type, start, end)
            gold_spans = {(e.type, e.start, e.end): e for e in gold.entities}
            pred_spans = {(e.type, e.start, e.end): e for e in pred.entities}

            g_keys = set(gold_spans.keys())
            p_keys = set(pred_spans.keys())
            tp_keys = g_keys & p_keys
            fp_keys = p_keys - g_keys
            fn_keys = g_keys - p_keys

            strict_tp += len(tp_keys)
            strict_fp += len(fp_keys)
            strict_fn += len(fn_keys)

            lang_counts[gold.language]["tp"] += len(tp_keys)
            lang_counts[gold.language]["fp"] += len(fp_keys)
            lang_counts[gold.language]["fn"] += len(fn_keys)

            doctype_counts[gold.doc_type]["tp"] += len(tp_keys)
            doctype_counts[gold.doc_type]["fp"] += len(fp_keys)
            doctype_counts[gold.doc_type]["fn"] += len(fn_keys)

            for k in tp_keys:
                type_counts[k[0]]["tp"] += 1
                assertion_pairs.append((gold_spans[k].assertion, pred_spans[k].assertion))
            for k in fp_keys:
                type_counts[k[0]]["fp"] += 1
            for k in fn_keys:
                type_counts[k[0]]["fn"] += 1

            for pe in pred.entities:
                confs.append(max(0.0, min(1.0, float(pe.confidence))))
                corr_flags.append(1 if (pe.type, pe.start, pe.end) in g_keys else 0)

            # Lenient (overlap + same type)
            matched_gold_idx: set[int] = set()
            l_tp = 0
            for pe in pred.entities:
                found = False
                for gi, ge in enumerate(gold.entities):
                    if gi not in matched_gold_idx and pe.type == ge.type and _overlaps(pe.start, pe.end, ge.start, ge.end):
                        matched_gold_idx.add(gi)
                        found = True
                        break
                if found:
                    l_tp += 1
            lenient_tp += l_tp
            lenient_fp += max(0, len(pred.entities) - l_tp)
            lenient_fn += max(0, len(gold.entities) - len(matched_gold_idx))

            # Relations
            ent_by_id = {e.id: e for e in gold.entities}
            gold_rels = {
                (r.type, ent_by_id[r.head].start, ent_by_id[r.head].end, ent_by_id[r.tail].start, ent_by_id[r.tail].end)
                for r in gold.relations
                if r.head in ent_by_id and r.tail in ent_by_id
            }
            pred_rels = {
                (r.type, r.head_start, r.head_end, r.tail_start, r.tail_end) for r in pred.relations
            }
            rel_tp += len(gold_rels & pred_rels)
            rel_fp += len(pred_rels - gold_rels)
            rel_fn += len(gold_rels - pred_rels)

            # TIMEX
            gold_tx = {(t.type, t.start, t.end): t for t in gold.timex}
            pred_tx = {(t.type, t.start, t.end): t for t in pred.timex}
            tx_match = set(gold_tx.keys()) & set(pred_tx.keys())
            timex_tp += len(tx_match)
            timex_fp += len(set(pred_tx.keys()) - set(gold_tx.keys()))
            timex_fn += len(set(gold_tx.keys()) - set(pred_tx.keys()))
            for k in tx_match:
                if gold_tx[k].normalized is not None:
                    timex_norm_total += 1
                    if gold_tx[k].normalized == pred_tx[k].normalized:
                        timex_norm_correct += 1

        per_type = {t: _prf(c["tp"], c["fp"], c["fn"]) for t, c in sorted(type_counts.items())}
        macro_f1 = round(sum(v["f1"] for v in per_type.values()) / len(per_type), 6) if per_type else 0.0
        per_lang = {lang: _prf(c["tp"], c["fp"], c["fn"]) for lang, c in sorted(lang_counts.items())}
        per_doc_type = {dt: _prf(c["tp"], c["fp"], c["fn"]) for dt, c in sorted(doctype_counts.items())}

        assertion_acc, assertion_macro = _macro_f1_from_pairs(assertion_pairs, ASSERTION_CLASSES)
        doc_cls_acc, doc_cls_macro = _macro_f1_from_pairs(doc_cls_pairs, sorted(all_doc_labels))
        cal = compute_ece_and_brier(confs, corr_flags)

        return {
            "documents_evaluated": len(gold_docs),
            "strict_span": {**_prf(strict_tp, strict_fp, strict_fn), "macro_f1": macro_f1},
            "lenient_span": _prf(lenient_tp, lenient_fp, lenient_fn),
            "per_entity_type": per_type,
            "per_language": per_lang,
            "per_doc_type": per_doc_type,
            "assertion": {"accuracy": assertion_acc, "macro_f1": assertion_macro, "support": len(assertion_pairs)},
            "relations": _prf(rel_tp, rel_fp, rel_fn),
            "timex": {
                **_prf(timex_tp, timex_fp, timex_fn),
                "normalization_accuracy": round(timex_norm_correct / timex_norm_total, 6) if timex_norm_total > 0 else 0.0,
            },
            "doc_classification": {"accuracy": doc_cls_acc, "macro_f1": doc_cls_macro, "support": len(doc_cls_pairs)},
            "calibration": cal,
        }


def per_doc_strict_counts(
    gold_docs: list[AnnotatedDocument],
    predictions: list[DocumentPrediction],
) -> list[tuple[int, int, int]]:
    """Return per-document strict entity (tp, fp, fn) counts aligned to gold_docs."""
    pred_map = {p.doc_id: p for p in predictions}
    counts: list[tuple[int, int, int]] = []
    for g in gold_docs:
        gold_set = {(e.type, e.start, e.end) for e in g.entities}
        pred_doc = pred_map.get(
            g.doc_id, DocumentPrediction(doc_id=g.doc_id, language=g.language, doc_type=g.doc_type)
        )
        pred_set = {(e.type, e.start, e.end) for e in pred_doc.entities}
        tp = len(gold_set & pred_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        counts.append((tp, fp, fn))
    return counts


def _f1_from_counts_arr(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    denom = 2.0 * tp + fp + fn
    return np.where(denom > 0.0, (2.0 * tp) / np.where(denom > 0.0, denom, 1.0), 0.0)


def bootstrap_f1_ci(
    gold_docs: list[AnnotatedDocument],
    predictions: list[DocumentPrediction],
    *,
    n_resamples: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Vectorized document-level bootstrap 95% confidence interval for strict micro-F1."""
    if not gold_docs:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    arr = np.asarray(per_doc_strict_counts(gold_docs, predictions), dtype=np.float64)
    n = len(arr)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    sampled = arr[idx].sum(axis=1)
    scores = np.sort(_f1_from_counts_arr(sampled[:, 0], sampled[:, 1], sampled[:, 2]))
    lo = float(scores[max(0, int(0.025 * len(scores)))])
    hi = float(scores[min(len(scores) - 1, int(0.975 * len(scores)))])
    return {
        "mean": round(float(scores.mean()), 6),
        "ci_lower": round(lo, 6),
        "ci_upper": round(hi, 6),
    }


def bootstrap_mean_seed_f1_ci(
    doc_seed_counts: list[list[tuple[int, int, int]]],
    *,
    n_resamples: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Document-level bootstrap 95% CI for mean-across-seeds micro-F1.

    Resamples the N_docs independent documents (never N_docs * N_seeds pooled)
    and applies the exact same estimator (per-seed micro-F1 averaged over seeds)
    so the CI always centers on and contains the reported point estimate.
    """
    if not doc_seed_counts:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    arr = np.asarray(doc_seed_counts, dtype=np.float64)  # (N_docs, N_seeds, 3)
    n_docs = arr.shape[0]
    tot = arr.sum(axis=0)  # (N_seeds, 3)
    seed_f1s = _f1_from_counts_arr(tot[:, 0], tot[:, 1], tot[:, 2])
    point_est = float(seed_f1s.mean())

    rng = np.random.default_rng(seed)
    boot_vals = np.empty(n_resamples, dtype=np.float64)
    chunk = 2000
    for start in range(0, n_resamples, chunk):
        sz = min(chunk, n_resamples - start)
        idx = rng.integers(0, n_docs, size=(sz, n_docs))
        sampled = arr[idx].sum(axis=1)  # (sz, N_seeds, 3)
        denom = 2.0 * sampled[:, :, 0] + sampled[:, :, 1] + sampled[:, :, 2]
        f1s = np.where(denom > 0.0, (2.0 * sampled[:, :, 0]) / np.where(denom > 0.0, denom, 1.0), 0.0)
        boot_vals[start : start + sz] = f1s.mean(axis=1)

    sorted_b = np.sort(boot_vals)
    lo = float(sorted_b[max(0, int(0.025 * n_resamples))])
    hi = float(sorted_b[min(n_resamples - 1, int(0.975 * n_resamples))])
    return {
        "mean": round(point_est, 6),
        "ci_lower": round(lo, 6),
        "ci_upper": round(hi, 6),
    }


def holm_bonferroni_adjust(p_values: list[float]) -> list[float]:
    """Apply step-down Holm-Bonferroni family-wise error correction."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [1.0] * m
    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed):
        scaled = min(1.0, (m - rank) * float(p_val))
        running_max = max(running_max, scaled)
        adjusted[orig_idx] = running_max
    return adjusted


def pooled_paired_bootstrap(
    doc_seed_counts_a: list[list[tuple[int, int, int]]],
    doc_seed_counts_b: list[list[tuple[int, int, int]]],
    *,
    n_resamples: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Cluster-paired bootstrap over documents pooled across seeds (N_docs x N_seeds)."""
    if len(doc_seed_counts_a) != len(doc_seed_counts_b) or not doc_seed_counts_a:
        raise ValueError("doc_seed_counts_a and doc_seed_counts_b must be non-empty and equal length")
    arr_a = np.asarray(doc_seed_counts_a, dtype=np.float64).sum(axis=1)  # (N_docs, 3)
    arr_b = np.asarray(doc_seed_counts_b, dtype=np.float64).sum(axis=1)  # (N_docs, 3)
    n = len(arr_a)
    tot_a = arr_a.sum(axis=0)
    tot_b = arr_b.sum(axis=0)
    f1_a = float(_f1_from_counts_arr(tot_a[0:1], tot_a[1:2], tot_a[2:3])[0])
    f1_b = float(_f1_from_counts_arr(tot_b[0:1], tot_b[1:2], tot_b[2:3])[0])
    obs_delta = f1_b - f1_a

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_resamples, dtype=np.float64)
    chunk = 2000
    for start in range(0, n_resamples, chunk):
        sz = min(chunk, n_resamples - start)
        idx = rng.integers(0, n, size=(sz, n))
        sa = arr_a[idx].sum(axis=1)
        sb = arr_b[idx].sum(axis=1)
        fa = _f1_from_counts_arr(sa[:, 0], sa[:, 1], sa[:, 2])
        fb = _f1_from_counts_arr(sb[:, 0], sb[:, 1], sb[:, 2])
        deltas[start : start + sz] = fb - fa

    if obs_delta >= 0.0:
        extreme = int(np.sum(deltas <= 0.0))
    else:
        extreme = int(np.sum(deltas >= 0.0))
    p_val = min(1.0, 2.0 * (extreme + 1) / (n_resamples + 1))
    sorted_d = np.sort(deltas)
    lo = float(sorted_d[max(0, int(0.025 * len(sorted_d)))])
    hi = float(sorted_d[min(len(sorted_d) - 1, int(0.975 * len(sorted_d)))])

    return {
        "f1_a": round(f1_a, 6),
        "f1_b": round(f1_b, 6),
        "delta_f1": round(obs_delta, 6),
        "ci95_low": round(lo, 6),
        "ci95_high": round(hi, 6),
        "p_value": round(p_val, 8),
        "n_resamples": n_resamples,
    }


def paired_significance_test(
    gold_docs: list[AnnotatedDocument],
    preds_a: list[DocumentPrediction],
    preds_b: list[DocumentPrediction],
    *,
    n_resamples: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired bootstrap test (delta F1 = F1_B - F1_A) and McNemar test on gold entity hits."""
    map_a = {p.doc_id: p for p in preds_a}
    map_b = {p.doc_id: p for p in preds_b}
    counts_a = [[c] for c in per_doc_strict_counts(gold_docs, preds_a)]
    counts_b = [[c] for c in per_doc_strict_counts(gold_docs, preds_b)]
    boot = pooled_paired_bootstrap(counts_a, counts_b, n_resamples=n_resamples, seed=seed)

    # McNemar contingency counts on gold entity span hits
    n10 = n01 = 0
    for g in gold_docs:
        pa = {(e.type, e.start, e.end) for e in map_a.get(g.doc_id, DocumentPrediction(g.doc_id, g.language, g.doc_type)).entities}
        pb = {(e.type, e.start, e.end) for e in map_b.get(g.doc_id, DocumentPrediction(g.doc_id, g.language, g.doc_type)).entities}
        for ge in g.entities:
            key = (ge.type, ge.start, ge.end)
            hit_a = key in pa
            hit_b = key in pb
            if hit_a and not hit_b:
                n10 += 1
            elif hit_b and not hit_a:
                n01 += 1

    # Continuity-corrected McNemar chi-square p-value (1 dof)
    if n10 + n01 == 0:
        mcnemar_p = 1.0
    else:
        chi2 = ((abs(n10 - n01) - 1.0) ** 2) / (n10 + n01)
        mcnemar_p = math.erfc(math.sqrt(chi2 / 2.0))

    return {
        "f1_a": boot["f1_a"],
        "f1_b": boot["f1_b"],
        "delta_f1": boot["delta_f1"],
        "bootstrap_p_value": boot["p_value"],
        "n_resamples": n_resamples,
        "mcnemar_n10": n10,
        "mcnemar_n01": n01,
        "mcnemar_p_value": round(mcnemar_p, 8),
    }
