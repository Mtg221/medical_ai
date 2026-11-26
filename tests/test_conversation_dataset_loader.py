"""Validation & conversion tests for the bilingual EN/FR medical conversation dataset loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from medresearch_platform.data.conversation_loader import (
    _convert_conversation_record,
    load_conversation_dataset_from_zip,
)

ZIP_PATH = Path("medical_conversation_dataset_en_fr.zip")


def test_converter_rejects_invalid_offsets_and_maps_family_and_relations() -> None:
    bad_rec = {
        "conversation_id": "BAD_01",
        "split": "train",
        "language": "en",
        "text": "Patient has headache.",
        "spans": [{"entity_id": "E1", "char_start": 0, "char_end": 5, "text": "WRONG", "entity_type": "SYMPTOM"}],
        "annotation": {},
    }
    doc, err = _convert_conversation_record(bad_rec)
    assert doc is None and err is not None and "span_offset_mismatch" in err

    good_rec = {
        "conversation_id": "GOOD_01",
        "encounter_id": "ENC_1",
        "split": "train",
        "language": "en",
        "noise_severity": "moderate",
        "text": "Mother has migraine. Ibuprofen 400 mg PO.",
        "spans": [
            {"entity_id": "E1", "char_start": 11, "char_end": 19, "text": "migraine", "entity_type": "FAMILY_HISTORY", "assertion": "present", "experiencer": "family", "temporality": "historical"},
            {"entity_id": "E2", "char_start": 21, "char_end": 30, "text": "Ibuprofen", "entity_type": "MEDICATION", "assertion": "present", "experiencer": "patient", "temporality": "current"},
            {"entity_id": "E3", "char_start": 31, "char_end": 37, "text": "400 mg", "entity_type": "DOSAGE", "assertion": "present", "experiencer": "patient", "temporality": "current"},
        ],
        "annotation": {
            "relations": [
                {"relation_id": "R1", "relation_type": "medication_has_dose", "head_entity_id": "E2", "tail_entity_id": "E3"}
            ]
        },
    }
    doc_ok, err_ok = _convert_conversation_record(good_rec)
    assert err_ok is None and doc_ok is not None
    assert doc_ok.validate_spans() == []
    assert doc_ok.entities[0].assertion == "family"
    assert len(doc_ok.relations) == 1 and doc_ok.relations[0].type == "HAS_DOSAGE"


def test_load_full_conversation_archive_zero_leakage() -> None:
    if not ZIP_PATH.exists():
        pytest.skip("medical_conversation_dataset_en_fr.zip not present in fresh-clone CI")
    res = load_conversation_dataset_from_zip(ZIP_PATH)
    assert res.audit["total_loaded"] == 1200
    assert res.audit["excluded_n"] == 0
    assert res.audit["train_n"] == 960
    assert res.audit["validation_n"] == 122
    assert res.audit["test_n"] == 118
    assert res.audit["leakage"]["train_val_encounter_overlap"] == 0
    assert res.audit["leakage"]["train_test_encounter_overlap"] == 0
    assert res.audit["leakage"]["val_test_encounter_overlap"] == 0
