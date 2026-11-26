"""Strict protocol enforcement tests for the 80-note human realistic evaluation tooling (Phase 2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from medresearch_platform.data.schema import DOC_TYPES
from scripts.eval_realistic import verify_owner_freeze_manifest
from scripts.import_label_studio_realistic import (
    build_freeze_manifest,
    compute_label_studio_iaa,
    convert_label_studio_records,
)


def _make_task(
    idx: int,
    *,
    lang: str,
    doc_type: str,
    double_annotated: bool = False,
    disagree: bool = False,
    adjudicated: bool = False,
    omit_field: str | None = None,
    text_override: str | None = None,
) -> dict[str, Any]:
    text = text_override or (
        f"Patient {idx} presents with fever on 2025-03-10."
        if lang == "en"
        else f"Patient {idx} présente de la fièvre le 2025-03-10."
    )
    symptom = "fever" if lang == "en" else "fièvre"
    s = text.index(symptom)
    e = s + len(symptom)
    ann_a = {
        "id": idx * 10 + 1,
        "result": [
            {"id": f"r{idx}a", "type": "labels", "from_name": "entities", "value": {"start": s, "end": e, "labels": ["SYMPTOM"]}},
        ],
    }
    annotations: list[dict[str, Any]] = [ann_a]
    if double_annotated:
        ann_b = {
            "id": idx * 10 + 2,
            "result": [
                {
                    "id": f"r{idx}b",
                    "type": "labels",
                    "from_name": "entities",
                    "value": {"start": s, "end": e, "labels": ["DIAGNOSIS" if disagree else "SYMPTOM"]},
                }
            ],
        }
        annotations.append(ann_b)
        if adjudicated:
            annotations.append(
                {
                    "id": idx * 10 + 3,
                    "ground_truth": True,
                    "result": [
                        {"id": f"r{idx}g", "type": "labels", "from_name": "entities", "value": {"start": s, "end": e, "labels": ["SYMPTOM"]}},
                    ],
                }
            )
    data: dict[str, Any] = {
        "doc_id": f"fixture_{lang}_{idx:03d}",
        "language": lang,
        "doc_type": doc_type,
        "document_date": "2025-03-15",
        "text": text,
    }
    if omit_field is not None:
        data.pop(omit_field, None)
    return {"id": idx, "data": data, "annotations": annotations}


def _make_valid_80_fixture(*, double_count: int = 16, disagree_without_adj: bool = False) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for i in range(40):
        is_dbl = i < (double_count // 2)
        tasks.append(
            _make_task(
                i + 1,
                lang="en",
                doc_type=DOC_TYPES[i % len(DOC_TYPES)],
                double_annotated=is_dbl,
                disagree=is_dbl and (i == 0),
                adjudicated=is_dbl and (i == 0) and not disagree_without_adj,
            )
        )
    for i in range(40):
        is_dbl = i < (double_count - (double_count // 2))
        tasks.append(
            _make_task(
                41 + i,
                lang="fr",
                doc_type=DOC_TYPES[i % len(DOC_TYPES)],
                double_annotated=is_dbl,
                disagree=is_dbl and (i == 0),
                adjudicated=is_dbl and (i == 0) and not disagree_without_adj,
            )
        )
    return tasks


@pytest.mark.parametrize("missing_field", ["doc_id", "language", "doc_type", "document_date"])
def test_strict_import_rejects_missing_explicit_metadata(missing_field: str) -> None:
    task = _make_task(1, lang="en", doc_type="consultation", omit_field=missing_field)
    with pytest.raises(ValueError, match=missing_field):
        convert_label_studio_records([task], require_explicit_metadata=True)


def test_strict_import_rejects_unadjudicated_disagreement() -> None:
    task = _make_task(1, lang="en", doc_type="consultation", double_annotated=True, disagree=True, adjudicated=False)
    with pytest.raises(ValueError, match="unadjudicated disagreement"):
        convert_label_studio_records([task], require_adjudication=True)


def test_strict_import_accepts_explicitly_adjudicated_gold_and_computes_iaa_on_independent() -> None:
    task = _make_task(1, lang="en", doc_type="consultation", double_annotated=True, disagree=True, adjudicated=True)
    docs = convert_label_studio_records([task], require_explicit_metadata=True, require_adjudication=True)
    assert len(docs) == 1
    assert docs[0].entities[0].type == "SYMPTOM"
    iaa = compute_label_studio_iaa([task], require_explicit_metadata=True)
    assert iaa["double_annotated_docs"] == 1
    # Independent annotators disagreed (SYMPTOM vs DIAGNOSIS), so entity_strict_f1 == 0.0
    assert iaa["entity_strict_f1"] == 0.0


def test_strict_protocol_rejects_incomplete_or_imbalanced_sets() -> None:
    # Too few docs
    short_set = _make_valid_80_fixture()[:10]
    with pytest.raises(ValueError, match="Expected exactly 80 documents"):
        convert_label_studio_records(short_set, strict_protocol=True)

    # Too few double-annotated docs (< 16)
    low_double = _make_valid_80_fixture(double_count=10)
    with pytest.raises(ValueError, match=">= 16 double-annotated"):
        convert_label_studio_records(low_double, strict_protocol=True)

    # Unadjudicated disagreement inside full 80-doc set
    unadj = _make_valid_80_fixture(double_count=16, disagree_without_adj=True)
    with pytest.raises(ValueError, match="unadjudicated disagreement"):
        convert_label_studio_records(unadj, strict_protocol=True)

    # Missing a doc_type
    missing_type = _make_valid_80_fixture()
    for t in missing_type:
        if t["data"]["doc_type"] == "operative_note":
            t["data"]["doc_type"] = "consultation"
    with pytest.raises(ValueError, match="Missing required document types"):
        convert_label_studio_records(missing_type, strict_protocol=True)

    # Duplicate normalized text
    dup_text = _make_valid_80_fixture()
    dup_text[5]["data"]["text"] = dup_text[4]["data"]["text"]
    with pytest.raises(ValueError, match="duplicate normalized document text"):
        convert_label_studio_records(dup_text, strict_protocol=True)


def test_strict_protocol_passes_valid_80_fixture_and_freeze_manifest_detects_tampering(tmp_path: Path) -> None:
    tasks = _make_valid_80_fixture(double_count=16)
    docs = convert_label_studio_records(tasks, source="test_fixture_only", strict_protocol=True)
    iaa = compute_label_studio_iaa(tasks, source="test_fixture_only", require_explicit_metadata=True)
    assert len(docs) == 80
    assert iaa["double_annotated_docs"] == 16

    in_path = tmp_path / "export.json"
    out_path = tmp_path / "eval.jsonl"
    manifest_path = tmp_path / "eval.jsonl.manifest.json"
    in_path.write_text(json.dumps(tasks), encoding="utf-8")
    out_path.write_text("\n".join(json.dumps(d.to_dict()) for d in docs) + "\n", encoding="utf-8")

    manifest = build_freeze_manifest(
        input_path=in_path,
        output_path=out_path,
        docs=docs,
        iaa=iaa,
        source="test_fixture_only",
        dataset_version="fixture-v1.0",
        protocol_validated=True,
        adjudicated_docs=2,
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded_manifest, _ = verify_owner_freeze_manifest(out_path, docs, manifest_path=manifest_path)
    assert loaded_manifest["protocol_validated"] is True
    assert loaded_manifest["double_annotated_docs"] == 16

    # Tamper with frozen eval.jsonl -> must be rejected
    out_path.write_text(out_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_owner_freeze_manifest(out_path, docs, manifest_path=manifest_path)
