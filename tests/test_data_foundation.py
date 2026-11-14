"""Tests for Phase 1 Data Foundation: schema, converters, synthetic v2, splits, and leakage."""

from __future__ import annotations

import io
import random
from pathlib import Path

import yaml
from PIL import Image

from medresearch_platform.data.builder import build_corpus_from_config
from medresearch_platform.data.converters import (
    from_brat_standoff,
    from_label_studio,
    summarize_label_mapping,
    to_brat_standoff,
    to_label_studio,
)
from medresearch_platform.data.schema import DOC_TYPES
from medresearch_platform.data.splits import assign_stratified_splits, verify_no_split_leakage
from medresearch_platform.data.synthetic_v2 import (
    generate_synthetic_corpus,
    generate_synthetic_document,
    render_degraded_scan_png,
)


def test_synthetic_v2_span_integrity_en_and_fr() -> None:
    docs = generate_synthetic_corpus(en_count=18, fr_count=18, seed=123)
    assert len(docs) == 36
    assert {d.language for d in docs} == {"en", "fr"}
    assert {d.doc_type for d in docs} == set(DOC_TYPES)

    assertions_seen = set()
    for d in docs:
        assert d.validate_spans() == []
        for e in d.entities:
            assertions_seen.add(e.assertion)
    assert {"present", "absent", "possible", "hypothetical", "family", "historical"} <= assertions_seen


def test_brat_and_label_studio_roundtrips() -> None:
    rng = random.Random(7)
    doc = generate_synthetic_document(
        doc_index=1, patient_index=1, language="fr", doc_type="discharge_summary", rng=rng
    )
    txt, ann = to_brat_standoff(doc)
    reimported = from_brat_standoff(
        txt,
        ann,
        doc_id=doc.doc_id,
        patient_id=doc.patient_id,
        language=doc.language,
        doc_type=doc.doc_type,
        document_date=doc.document_date,
    )
    assert reimported.validate_spans() == []
    assert len(reimported.entities) == len(doc.entities)
    assert len(reimported.relations) == len(doc.relations)

    ls_task = to_label_studio(doc)
    ls_doc = from_label_studio(ls_task)
    assert ls_doc.validate_spans() == []
    assert len(ls_doc.entities) == len(doc.entities)


def test_external_label_mapping_and_unmapped_reporting() -> None:
    rep = summarize_label_mapping("quaero", ["DISO", "CHEM", "PROC", "ANAT", "GEOG"])
    assert rep["total"] == 5
    assert rep["mapped"]["DISO->DIAGNOSIS"] == 1
    assert rep["unmapped"] == {"GEOG": 1}


def test_stratified_splits_and_leakage_detection() -> None:
    docs = generate_synthetic_corpus(en_count=30, fr_count=30, seed=42)
    assign_stratified_splits(docs, seed=42)
    res = verify_no_split_leakage(docs, minhash_threshold=0.85)
    assert res["passed"] is True
    assert res["shared_patients"] == []
    assert res["near_duplicates"] == []

    # Inject a near-duplicate across splits and confirm leakage detector catches it
    train_doc = next(d for d in docs if d.split == "train")
    test_doc = next(d for d in docs if d.split == "test")
    test_doc.text = train_doc.text
    leaked = verify_no_split_leakage(docs, minhash_threshold=0.85)
    assert leaked["passed"] is False
    assert len(leaked["near_duplicates"]) >= 1


def test_degraded_scan_png_and_corpus_builder(tmp_path: Path) -> None:
    png_bytes = render_degraded_scan_png("Patient presents with chest pain.\nHbA1c: 7.2 %", seed=9)
    with Image.open(io.BytesIO(png_bytes)) as img:
        assert img.format == "PNG"
        assert img.width == 1000

    cfg_path = tmp_path / "data.yaml"
    out_dir = tmp_path / "processed"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "dataset_name": "SYNTH-TEST",
                "version": "1.0.0",
                "seed": 42,
                "output_dir": str(out_dir),
                "languages": {"en": 20, "fr": 20},
                "splits": {"train": 0.70, "val": 0.15, "test": 0.15},
                "render_samples": {"enabled": True, "count_per_language": 1, "typo_rate": 0.05},
            }
        ),
        encoding="utf-8",
    )
    manifest = build_corpus_from_config(cfg_path)
    assert manifest["stats"]["total_documents"] == 40
    assert manifest["leakage_check"]["passed"] is True
    assert manifest["test_split_locked"] is True
    assert (out_dir / "train.jsonl").exists()
    assert (out_dir / "val.jsonl").exists()
    assert (out_dir / "test.jsonl").exists()
    assert (out_dir / "manifest.json").exists()
