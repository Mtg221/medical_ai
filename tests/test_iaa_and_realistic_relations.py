"""Tests for Label Studio relation/TIMEX import, double-annotation IAA, and human PHI redaction eval."""

from __future__ import annotations

from pathlib import Path

import pytest

from medresearch_platform.eval_harness.iaa import cohens_kappa
from scripts.eval_redaction_human import evaluate_redaction_jsonl
from scripts.import_label_studio_realistic import (
    compute_label_studio_iaa,
    convert_label_studio_records,
)


def _sample_double_annotated_tasks() -> list[dict]:
    en_text = "Started metformin 500 mg BID on 2025-03-10 for diabetes; denies fever yesterday."
    fr_text = "Début d'amoxicilline 1 g 2x/j le 1er mars 2025 pour pneumonie ; pas de dyspnée."
    return [
        {
            "id": 1,
            "data": {
                "doc_id": "owner_en_01",
                "language": "en",
                "doc_type": "consultation",
                "document_date": "2025-03-15",
                "text": en_text,
            },
            "annotations": [
                {
                    "id": 101,
                    "result": [
                        {"id": "r1", "type": "labels", "from_name": "entities", "value": {"start": 8, "end": 17, "labels": ["MEDICATION"]}},
                        {"id": "r2", "type": "labels", "from_name": "entities", "value": {"start": 18, "end": 24, "labels": ["DOSAGE"]}},
                        {"id": "r3", "type": "labels", "from_name": "entities", "value": {"start": 25, "end": 28, "labels": ["FREQUENCY"]}},
                        {"id": "r4", "type": "labels", "from_name": "entities", "value": {"start": 32, "end": 42, "labels": ["DATE"]}},
                        {"id": "r5", "type": "labels", "from_name": "entities", "value": {"start": 47, "end": 55, "labels": ["DIAGNOSIS"]}},
                        {"id": "r6", "type": "labels", "from_name": "entities", "value": {"start": 64, "end": 69, "labels": ["SYMPTOM"]}},
                        {"id": "r6", "type": "choices", "from_name": "assertion", "value": {"choices": ["absent"]}},
                        {"id": "t1", "type": "labels", "from_name": "timex", "value": {"start": 70, "end": 79, "labels": ["RELATIVE"]}},
                        {"type": "relation", "from_id": "r1", "to_id": "r2", "labels": ["HAS_DOSAGE"]},
                        {"type": "relation", "from_id": "r1", "to_id": "r5", "labels": ["TREATS"]},
                    ],
                },
                {
                    "id": 102,
                    "result": [
                        {"id": "b1", "type": "labels", "from_name": "entities", "value": {"start": 8, "end": 17, "labels": ["MEDICATION"]}},
                        {"id": "b2", "type": "labels", "from_name": "entities", "value": {"start": 18, "end": 24, "labels": ["DOSAGE"]}},
                        {"id": "b3", "type": "labels", "from_name": "entities", "value": {"start": 25, "end": 28, "labels": ["FREQUENCY"]}},
                        {"id": "b4", "type": "labels", "from_name": "entities", "value": {"start": 32, "end": 42, "labels": ["DATE"]}},
                        {"id": "b5", "type": "labels", "from_name": "entities", "value": {"start": 47, "end": 55, "labels": ["DIAGNOSIS"]}},
                        {"id": "b6", "type": "labels", "from_name": "entities", "value": {"start": 64, "end": 69, "labels": ["SYMPTOM"]}},
                        {"id": "b6", "type": "choices", "from_name": "assertion", "value": {"choices": ["absent"]}},
                        {"id": "bt1", "type": "labels", "from_name": "timex", "value": {"start": 70, "end": 79, "labels": ["RELATIVE"]}},
                        {"type": "relation", "from_id": "b1", "to_id": "b2", "labels": ["HAS_DOSAGE"]},
                    ],
                },
            ],
        },
        {
            "id": 2,
            "data": {
                "doc_id": "owner_fr_01",
                "language": "fr",
                "doc_type": "prescription",
                "document_date": "2025-03-15",
                "text": fr_text,
            },
            "annotations": [
                {
                    "id": 201,
                    "result": [
                        {"id": "f1", "type": "labels", "from_name": "entities", "value": {"start": 8, "end": 20, "labels": ["MEDICATION"]}},
                        {"id": "f2", "type": "labels", "from_name": "entities", "value": {"start": 21, "end": 24, "labels": ["DOSAGE"]}},
                        {"id": "f3", "type": "labels", "from_name": "entities", "value": {"start": 25, "end": 29, "labels": ["FREQUENCY"]}},
                        {"id": "f4", "type": "labels", "from_name": "entities", "value": {"start": 30, "end": 46, "labels": ["DATE"]}},
                        {"id": "f5", "type": "labels", "from_name": "entities", "value": {"start": 52, "end": 61, "labels": ["DIAGNOSIS"]}},
                        {"id": "f6", "type": "labels", "from_name": "entities", "value": {"start": 71, "end": 78, "labels": ["SYMPTOM"]}},
                        {"id": "f6", "type": "choices", "from_name": "assertion", "value": {"choices": ["absent"]}},
                        {"type": "relation", "from_id": "f1", "to_id": "f5", "labels": ["TREATS"]},
                    ],
                }
            ],
        },
    ]


def test_import_parses_relations_and_normalized_timex() -> None:
    tasks = _sample_double_annotated_tasks()
    docs = convert_label_studio_records(tasks)
    assert len(docs) == 2

    en_doc, fr_doc = docs
    assert [(r.type, r.head, r.tail) for r in en_doc.relations] == [
        ("HAS_DOSAGE", "e1", "e2"),
        ("TREATS", "e1", "e5"),
    ]
    assert {(t.type, t.text, t.normalized) for t in en_doc.timex} == {
        ("FREQUENCY", "BID", "BID"),
        ("DATE", "2025-03-10", "2025-03-10"),
        ("RELATIVE", "yesterday", "2025-03-14"),
    }
    assert {(t.type, t.text, t.normalized) for t in fr_doc.timex} == {
        ("FREQUENCY", "2x/j", "BID"),
        ("DATE", "le 1er mars 2025", "2025-03-01"),
    }
    assert en_doc.validate_spans() == []
    assert fr_doc.validate_spans() == []


def test_import_rejects_unknown_relation_and_dangling_region() -> None:
    bad_rel = [
        {
            "id": 9,
            "data": {"doc_id": "bad_rel", "language": "en", "text": "metformin 500 mg"},
            "annotations": [
                {
                    "result": [
                        {"id": "r1", "type": "labels", "value": {"start": 0, "end": 9, "labels": ["MEDICATION"]}},
                        {"id": "r2", "type": "labels", "value": {"start": 10, "end": 16, "labels": ["DOSAGE"]}},
                        {"type": "relation", "from_id": "r1", "to_id": "r2", "labels": ["INVENTED_REL"]},
                    ]
                }
            ],
        }
    ]
    with pytest.raises(ValueError, match="Unknown relation label 'INVENTED_REL'"):
        convert_label_studio_records(bad_rel)


def test_double_annotation_iaa_metrics() -> None:
    assert cohens_kappa(["present", "absent", "present", "possible"], ["present", "absent", "present", "possible"]) == 1.0
    iaa = compute_label_studio_iaa(_sample_double_annotated_tasks())
    assert iaa["total_tasks"] == 2
    assert iaa["double_annotated_docs"] == 1
    assert iaa["double_annotation_rate"] == 0.5
    assert iaa["entity_strict_f1"] == 1.0
    assert iaa["assertion_cohens_kappa"] == 1.0
    assert iaa["relation_strict_f1"] == pytest.approx(2 / 3, abs=1e-4)
    assert iaa["timex_strict_f1"] == 1.0
    assert "en" in iaa["per_language"]


def test_human_phi_redaction_runner_on_template() -> None:
    template = Path("docs/PHI_REDACTION_OWNER_TEMPLATE.jsonl")
    assert template.exists()
    report = evaluate_redaction_jsonl(template, owner_written=False)
    assert report["owner_written"] is False
    assert report["n_sentences"] == 4
    assert report["n_phi_items"] == 6
    assert report["recall"] == 1.0
    assert report["strict_full_token_recall"] == 1.0
    assert report["precision"] == 1.0
    assert report["fp_on_non_phi_sentences"] == 0
