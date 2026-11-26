"""Tests for staging medical_ai_eval_starter.zip into Label Studio draft tasks and PHI review JSONL."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from medresearch_platform.data.schema import ENTITY_TYPES
from scripts.prepare_eval_starter_for_label_studio import stage_eval_starter


def test_stage_eval_starter_maps_labels_offsets_and_uses_predictions_not_annotations(tmp_path: Path) -> None:
    zpath = tmp_path / "starter.zip"
    notes = [
        {
            "note_id": "CN001",
            "language": "en",
            "text": "The patient reports two weeks of Type 2 diabetes symptoms.",
            "entities": [
                {"text": "Type 2 diabetes", "label": "DISEASE", "assertion": "present"},
                {"text": "two weeks", "label": "DURATION"},
            ],
            "relations": [],
            "review_status": "draft_ai_assisted",
        },
        {
            "note_id": "CN002",
            "language": "fr",
            "text": "Pas de fièvre depuis trois jours.",
            "entities": [
                {"text": "fièvre", "label": "SYMPTOM", "assertion": "absent"},
                {"text": "trois jours", "label": "DURATION"},
            ],
            "relations": [],
            "review_status": "draft_ai_assisted",
        },
    ]
    phis = [
        {
            "sentence_id": "PHI-EN-01",
            "language": "en",
            "text": "Patient Jordan Reed attended the clinic today.",
            "entities": [{"text": "Jordan Reed", "label": "NAME"}],
            "source": "fictional_ai_assisted_draft",
            "review_status": "draft_ai_assisted",
        }
    ]
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("clinical_notes_80_draft.jsonl", "\n".join(json.dumps(n) for n in notes) + "\n")
        zf.writestr("phi_sentences_60_draft.jsonl", "\n".join(json.dumps(p) for p in phis) + "\n")

    out_dir = tmp_path / "staged"
    rep = stage_eval_starter(zpath, out_dir, train_paths=[])
    assert rep["clinical_notes_80"]["n_tasks"] == 2
    assert rep["clinical_notes_80"]["unmatched_spans"] == 0
    assert rep["phi_sentences_60"]["n_sentences"] == 1
    assert rep["phi_sentences_60"]["unmatched_spans"] == 0

    ls_tasks = json.loads((out_dir / "label_studio_tasks_80_draft.json").read_text(encoding="utf-8"))
    for task in ls_tasks:
        # Must store draft pre-labels in `predictions`, NEVER finalized `annotations`
        assert task["annotations"] == []
        assert len(task["predictions"]) == 1
        text = task["data"]["text"]
        for item in task["predictions"][0]["result"]:
            if item["type"] == "labels":
                v = item["value"]
                assert v["labels"][0] in set(ENTITY_TYPES)
                assert text[v["start"] : v["end"]] == v["text"]

    phi_rows = [
        json.loads(line)
        for line in (out_dir / "phi_60_draft_for_human_review.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert phi_rows[0]["phi_items"][0]["type"] == "PERSON"
    s = phi_rows[0]["phi_items"][0]["start"]
    e = phi_rows[0]["phi_items"][0]["end"]
    assert phi_rows[0]["text"][s:e] == "Jordan Reed"


def test_stage_eval_starter_blocks_zip_path_traversal(tmp_path: Path) -> None:
    zpath = tmp_path / "evil.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError, match="Blocked unsafe zip path traversal"):
        stage_eval_starter(zpath, tmp_path / "staged", train_paths=[])
