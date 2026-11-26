"""Tests for strict Label Studio import schema validation (rejects unknown labels like PROBLEM/TEST)."""

from __future__ import annotations

import pytest

from scripts.import_label_studio_realistic import convert_label_studio_records


def test_import_rejects_unknown_labels_like_problem_or_test() -> None:
    bad_task = [
        {
            "id": 1,
            "data": {"doc_id": "owner_en_01", "language": "en", "text": "Patient has pneumonia."},
            "annotations": [
                {
                    "result": [
                        {
                            "id": "r1",
                            "type": "labels",
                            "value": {"start": 12, "end": 21, "labels": ["PROBLEM"]},
                        }
                    ]
                }
            ],
        }
    ]
    with pytest.raises(ValueError, match="Unknown entity label 'PROBLEM'"):
        convert_label_studio_records(bad_task)

    bad_test_task = [
        {
            "id": 2,
            "data": {"doc_id": "owner_en_02", "language": "en", "text": "Ordered troponin."},
            "annotations": [
                {
                    "result": [
                        {
                            "id": "r2",
                            "type": "labels",
                            "value": {"start": 8, "end": 16, "labels": ["TEST"]},
                        }
                    ]
                }
            ],
        }
    ]
    with pytest.raises(ValueError, match="Unknown entity label 'TEST'"):
        convert_label_studio_records(bad_test_task)


def test_import_accepts_canonical_labels() -> None:
    good_task = [
        {
            "id": 1,
            "data": {"doc_id": "owner_en_03", "language": "en", "text": "No fever; ordered troponin."},
            "annotations": [
                {
                    "result": [
                        {"id": "r1", "type": "labels", "value": {"start": 3, "end": 8, "labels": ["SYMPTOM"]}},
                        {"id": "r1", "type": "choices", "from_name": "assertion", "value": {"choices": ["absent"]}},
                        {"id": "r2", "type": "labels", "value": {"start": 18, "end": 26, "labels": ["LAB_TEST"]}},
                    ]
                }
            ],
        }
    ]
    docs = convert_label_studio_records(good_task)
    assert len(docs) == 1
    assert [(e.type, e.text, e.assertion) for e in docs[0].entities] == [
        ("SYMPTOM", "fever", "absent"),
        ("LAB_TEST", "troponin", "present"),
    ]
