from __future__ import annotations

import inspect

import pytest

from medresearch_platform.data.schema import AnnotatedDocument, EntitySpan
from medresearch_platform.eval_harness.metrics import (
    DocumentPrediction,
    PredictedEntity,
    bootstrap_f1_ci,
    bootstrap_mean_seed_f1_ci,
    holm_bonferroni_adjust,
    paired_significance_test,
    pooled_paired_bootstrap,
)


def _make_doc(doc_id: str, n_entities: int) -> AnnotatedDocument:
    text = " ".join(f"term{i}" for i in range(n_entities))
    ents = []
    pos = 0
    for i in range(n_entities):
        tok = f"term{i}"
        ents.append(EntitySpan(id=f"e{i}", type="DIAGNOSIS", start=pos, end=pos + len(tok), text=tok, assertion="present"))
        pos += len(tok) + 1
    return AnnotatedDocument(
        doc_id=doc_id,
        patient_id="P1",
        document_date="2025-01-01",
        source="synthetic",
        language="en",
        doc_type="discharge_summary",
        text=text,
        entities=ents,
    )


def _make_pred(doc: AnnotatedDocument, hit_count: int) -> DocumentPrediction:
    return DocumentPrediction(
        doc_id=doc.doc_id,
        language=doc.language,
        doc_type=doc.doc_type,
        entities=[
            PredictedEntity(type=e.type, start=e.start, end=e.end, text=e.text, assertion=e.assertion)
            for e in doc.entities[:hit_count]
        ],
    )


def test_default_n_resamples_at_least_10000() -> None:
    sig_ci = inspect.signature(bootstrap_f1_ci)
    sig_pair = inspect.signature(paired_significance_test)
    assert sig_ci.parameters["n_resamples"].default >= 10000
    assert sig_pair.parameters["n_resamples"].default >= 10000


def test_holm_bonferroni_adjust_exact() -> None:
    raw = [0.001, 0.04, 0.03, 0.002]
    adj = holm_bonferroni_adjust(raw)
    # Sorted: 0.001 (*4=0.004), 0.002 (*3=0.006), 0.03 (*2=0.06), 0.04 (*1=0.06 via monotonicity)
    assert adj == pytest.approx([0.004, 0.06, 0.06, 0.006], abs=1e-9)


def test_paired_significance_10k_no_250_floor() -> None:
    docs = [_make_doc(f"d{i}", 4) for i in range(40)]
    preds_a = [_make_pred(d, 2) for d in docs]
    preds_b = [_make_pred(d, 4) for d in docs]
    res = paired_significance_test(docs, preds_a, preds_b, n_resamples=10000, seed=7)
    assert res["bootstrap_p_value"] < 0.001
    assert res["n_resamples"] == 10000


def test_pooled_paired_bootstrap_over_5_seeds() -> None:
    # 20 docs x 5 seeds per doc: model B consistently beats model A
    counts_a = [[(2, 0, 2) for _ in range(5)] for _ in range(20)]
    counts_b = [[(4, 0, 0) for _ in range(5)] for _ in range(20)]
    res = pooled_paired_bootstrap(counts_a, counts_b, n_resamples=10000, seed=42)
    assert res["delta_f1"] > 0.3
    assert res["p_value"] < 0.001
    assert res["n_resamples"] == 10000
    assert res["ci95_low"] < res["delta_f1"] < res["ci95_high"] or res["ci95_low"] == res["ci95_high"]


def test_bootstrap_mean_seed_f1_ci_contains_point_estimate_and_handles_deterministic() -> None:
    # Deterministic model across 5 seeds with per-doc variation (80 docs): CI must contain point estimate and have >0 width
    det_counts = [[(2 if i % 2 == 0 else 1, 1 if i % 3 == 0 else 0, 1) for _ in range(5)] for i in range(80)]
    res_det = bootstrap_mean_seed_f1_ci(det_counts, n_resamples=10000, seed=42)
    assert res_det["ci_lower"] <= res_det["mean"] <= res_det["ci_upper"]
    assert res_det["ci_upper"] > res_det["ci_lower"]

    # Stochastic model where seeds vary widely: CI still uses exact same mean-across-seeds estimator
    stoch_counts = [
        [(3 if (i + s) % 4 != 0 else 1, 1 if s == 0 else 0, 1 if i % 5 == 0 else 0) for s in range(5)]
        for i in range(160)
    ]
    res_stoch = bootstrap_mean_seed_f1_ci(stoch_counts, n_resamples=10000, seed=42)
    assert res_stoch["ci_lower"] <= res_stoch["mean"] <= res_stoch["ci_upper"]
