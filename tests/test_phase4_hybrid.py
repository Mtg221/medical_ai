"""Tests for Phase 4: 5-stage Hybrid pipeline, per-stage components, consistency checker, and CLI."""

from __future__ import annotations

import json
from pathlib import Path

from medresearch_platform.data.cli import main as cli_main
from medresearch_platform.data.synthetic_v2 import generate_synthetic_corpus
from medresearch_platform.eval_harness.metrics import (
    Evaluator,
    PredictedEntity,
    PredictedRelation,
    PredictedTimex,
)
from medresearch_platform.hybrid.pipeline import HybridPipeline
from medresearch_platform.hybrid.stage4_temporal import normalize_timex
from medresearch_platform.hybrid.stage5_integration import (
    TemperatureScaler,
    check_consistency,
)


def test_temporal_normalizer_and_consistency_checker() -> None:
    assert normalize_timex("3 days ago", "RELATIVE", "2026-03-10") == "2026-03-07"
    assert normalize_timex("il y a 5 jours", "RELATIVE", "2026-03-10") == "2026-03-05"
    assert normalize_timex("pendant 7 jours", "DURATION", "2026-03-10") == "P7D"

    entities = [
        PredictedEntity("DOSAGE", 0, 6, "500 mg", assertion="present", confidence=0.9),
        PredictedEntity("DIAGNOSIS", 10, 16, "asthma", assertion="present", confidence=0.9),
        PredictedEntity("DIAGNOSIS", 20, 26, "asthma", assertion="absent", confidence=0.9),
        PredictedEntity("LAB_TEST", 30, 35, "HbA1c", assertion="present", confidence=0.9),
        PredictedEntity("LAB_VALUE", 37, 41, "25.0", assertion="present", confidence=0.9),
        PredictedEntity("LAB_UNIT", 42, 46, "mg/L", assertion="present", confidence=0.9),
    ]
    rels = [
        PredictedRelation("HAS_VALUE", 30, 35, 37, 41),
        PredictedRelation("HAS_UNIT", 37, 41, 42, 46),
    ]
    timex = [PredictedTimex("DATE", 50, 60, "2026-04-01", normalized="2026-04-01")]
    warnings = check_consistency(entities, rels, timex, "2026-03-10")
    assert any("dose without medication" in w for w in warnings)
    assert any("conflicting assertions" in w for w in warnings)
    assert any("after document_date" in w for w in warnings)
    assert any("outside plausible range" in w for w in warnings)
    assert any("unit mismatch" in w for w in warnings)

    scaler = TemperatureScaler()
    t = scaler.fit([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    assert t > 0.0
    assert 0.0 < scaler.calibrate(0.8) < 1.0


def test_hybrid_pipeline_and_ablation_smoke() -> None:
    train_docs = generate_synthetic_corpus(en_count=20, fr_count=20, seed=11)
    val_docs = generate_synthetic_corpus(en_count=6, fr_count=6, seed=99)

    pipe = HybridPipeline(
        seed=42,
        cfg={"epochs": 6, "aux_epochs": 4, "hidden_dim": 24, "max_train_docs": 40},
    ).fit(train_docs)

    preds_full = []
    for d in val_docs:
        pred, contract = pipe.extract_document(d)
        preds_full.append(pred)
        assert contract["schema_version"] == "1.0"
        assert len(contract["processing"]["stages"]) == 5
        assert all("source" in e and "produced_by" in e for e in contract["entities"])

    ev = Evaluator()
    res_full = ev.evaluate(val_docs, preds_full)
    assert res_full["strict_span"]["f1"] > 0.50
    assert res_full["assertion"]["accuracy"] > 0.30
    assert res_full["timex"]["f1"] >= 0.20
    assert res_full["doc_classification"]["macro_f1"] > 0.30

    # Verify ablation toggle (no_rules_layer) executes cleanly
    preds_no_rules = [pipe.extract_document(d, use_rules_layer=False)[0] for d in val_docs]
    res_no_rules = ev.evaluate(val_docs, preds_no_rules)
    assert res_no_rules["strict_span"]["f1"] > 0.20


def test_cli_extract_command(tmp_path: Path, capsys) -> None:
    note = tmp_path / "sample.txt"
    note.write_text(
        "CLINICAL RECORD | Date: 2026-03-10\n"
        "CHIEF COMPLAINT:\nPatient presents with chest pain.\n"
        "MEDICATIONS:\nCurrent therapy: metformin 500 mg twice daily orally.\n",
        encoding="utf-8",
    )
    rc = cli_main(["extract", str(note), "--pipeline", "hybrid", "--train-jsonl", str(tmp_path / "missing.jsonl")])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["schema_version"] == "1.0"
    assert len(out["entities"]) >= 2
