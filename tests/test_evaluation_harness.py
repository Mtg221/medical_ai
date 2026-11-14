"""Tests for Phase 2 Evaluation Harness: exact hand-made metrics, taxonomy, profiler, and runner."""

from __future__ import annotations

from pathlib import Path

import yaml

from medresearch_platform.data.builder import build_corpus_from_config
from medresearch_platform.data.schema import (
    AnnotatedDocument,
    EntitySpan,
    RelationSpan,
    TimexSpan,
)
from medresearch_platform.eval_harness.error_taxonomy import classify_corpus_errors
from medresearch_platform.eval_harness.metrics import (
    DocumentPrediction,
    Evaluator,
    PredictedEntity,
    PredictedRelation,
    PredictedTimex,
    bootstrap_f1_ci,
    paired_significance_test,
)
from medresearch_platform.eval_harness.profiler import ResourceProfiler
from medresearch_platform.eval_harness.runner import run_experiment


def _handmade_docs() -> tuple[list[AnnotatedDocument], list[DocumentPrediction]]:
    text = "Patient has chest pain and takes metformin 500 mg since 2026-03-01."
    # Offsets:
    # "chest pain" -> 12:22
    # "metformin"  -> 33:42
    # "500 mg"     -> 43:49
    # "2026-03-01" -> 56:66
    gold = [
        AnnotatedDocument(
            doc_id="d1",
            patient_id="p1",
            language="en",
            doc_type="consultation",
            document_date="2026-03-10",
            source="handmade",
            text=text,
            entities=[
                EntitySpan("e1", "SYMPTOM", 12, 22, "chest pain", assertion="present"),
                EntitySpan("e2", "MEDICATION", 33, 42, "metformin", assertion="present"),
                EntitySpan("e3", "DOSAGE", 43, 49, "500 mg", assertion="present"),
            ],
            relations=[RelationSpan("r1", "HAS_DOSAGE", "e2", "e3")],
            timex=[TimexSpan("t1", "DATE", 56, 66, "2026-03-01", normalized="2026-03-01")],
        )
    ]
    # 2 strict TP (e1 with wrong assertion, e2 exact), 1 boundary error (e3 shifted 43:48), 1 spurious FP
    preds = [
        DocumentPrediction(
            doc_id="d1",
            language="en",
            doc_type="consultation",
            predicted_doc_type="consultation",
            entities=[
                PredictedEntity("SYMPTOM", 12, 22, "chest pain", assertion="absent", confidence=0.8),
                PredictedEntity("MEDICATION", 33, 42, "metformin", assertion="present", confidence=0.9),
                PredictedEntity("DOSAGE", 43, 48, "500 m", assertion="present", confidence=0.7),
                PredictedEntity("DIAGNOSIS", 0, 7, "Patient", assertion="present", confidence=0.4),
            ],
            relations=[PredictedRelation("HAS_DOSAGE", 33, 42, 43, 49, confidence=0.8)],
            timex=[PredictedTimex("DATE", 56, 66, "2026-03-01", normalized="2026-03-01")],
        )
    ]
    return gold, preds


def test_evaluator_handmade_exact_numbers() -> None:
    gold, preds = _handmade_docs()
    ev = Evaluator()
    res = ev.evaluate(gold, preds)

    # Strict: 2 TP, 2 FP, 1 FN -> P = 0.5, R = 2/3, F1 = 4/7
    assert res["strict_span"]["precision"] == 0.5
    assert abs(res["strict_span"]["recall"] - 0.666667) < 1e-5
    assert abs(res["strict_span"]["f1"] - 0.571429) < 1e-5

    # Lenient: e3 overlaps -> 3 TP, 1 FP, 0 FN -> P = 0.75, R = 1.0
    assert res["lenient_span"]["precision"] == 0.75
    assert res["lenient_span"]["recall"] == 1.0

    # Assertion: 1 of 2 matched spans correct -> 0.5
    assert res["assertion"]["accuracy"] == 0.5
    assert res["relations"]["f1"] == 1.0
    assert res["timex"]["f1"] == 1.0
    assert res["timex"]["normalization_accuracy"] == 1.0
    assert res["doc_classification"]["macro_f1"] == 1.0
    assert 0.0 <= res["calibration"]["brier"] <= 1.0

    ci = bootstrap_f1_ci(gold, preds, n_resamples=50, seed=42)
    assert ci["ci_lower"] <= ci["mean"] <= ci["ci_upper"]

    sig = paired_significance_test(gold, preds, preds, n_resamples=50, seed=42)
    assert sig["delta_f1"] == 0.0
    assert sig["mcnemar_p_value"] == 1.0


def test_error_taxonomy_and_profiler() -> None:
    gold, preds = _handmade_docs()
    errors = classify_corpus_errors(gold, preds)
    cats = {e.category for e in errors}
    assert {"assertion_wrong", "boundary_error", "spurious_fp"} <= cats

    with ResourceProfiler(docs_count=1, parameter_count=100) as prof:
        _ = sum(range(1000))
    assert prof.metrics is not None
    assert prof.metrics.wall_seconds > 0
    assert prof.metrics.peak_rss_mb > 0


def test_dummy_experiment_runner_end_to_end(tmp_path: Path) -> None:
    data_dir = tmp_path / "processed"
    data_cfg = tmp_path / "data.yaml"
    data_cfg.write_text(
        yaml.safe_dump(
            {
                "dataset_name": "SYNTH-E2E",
                "version": "1.0.0",
                "seed": 42,
                "output_dir": str(data_dir),
                "languages": {"en": 30, "fr": 30},
                "render_samples": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )
    build_corpus_from_config(data_cfg)

    exp_cfg = tmp_path / "dummy.yaml"
    runs_dir = tmp_path / "runs"
    exp_cfg.write_text(
        yaml.safe_dump(
            {
                "approach": "dummy",
                "eval_split": "val",
                "data_dir": str(data_dir),
                "output_root": str(runs_dir),
                "seeds": [1, 2],
            }
        ),
        encoding="utf-8",
    )
    res = run_experiment(exp_cfg)
    run_dir = runs_dir / res["run_id"]
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "errors.jsonl").exists()
    assert (run_dir / "resources.json").exists()
    assert (run_dir / "model_card.md").exists()
    assert res["summary"]["strict_f1"]["mean"] > 0.0
