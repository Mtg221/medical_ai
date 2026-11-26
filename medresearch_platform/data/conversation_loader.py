"""Validated loader & converter for the bilingual EN/FR medical conversation dataset archive."""

from __future__ import annotations

import json
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medresearch_platform.data.schema import (
    AnnotatedDocument,
    EntitySpan,
    RelationSpan,
    TimexSpan,
)

CONVERSATION_ENTITY_MAP: dict[str, str] = {
    "SYMPTOM": "SYMPTOM",
    "DIAGNOSIS": "DIAGNOSIS",
    "MEDICAL_HISTORY": "DIAGNOSIS",
    "FAMILY_HISTORY": "DIAGNOSIS",
    "ALLERGY": "DIAGNOSIS",
    "MEDICATION": "MEDICATION",
    "DOSAGE": "DOSAGE",
    "FREQUENCY": "FREQUENCY",
    "ROUTE": "ROUTE",
    "DURATION": "DURATION",
    "LAB_TEST": "LAB_TEST",
    "VITAL_SIGN": "LAB_TEST",
    "LAB_VALUE": "LAB_VALUE",
    "LAB_UNIT": "LAB_UNIT",
    "PROCEDURE": "PROCEDURE",
    "EXAMINATION": "PROCEDURE",
    "IMAGING_EXAM": "PROCEDURE",
    "ANATOMY": "ANATOMY",
    "DATE": "DATE",
}

CONVERSATION_RELATION_MAP: dict[str, str] = {
    "test_has_result": "HAS_VALUE",
    "test_has_unit": "HAS_UNIT",
    "medication_has_dose": "HAS_DOSAGE",
    "medication_has_route": "HAS_ROUTE",
    "medication_has_frequency": "HAS_FREQUENCY",
    "medication_treats_symptom": "TREATS",
    "symptom_has_location": "LOCATED_IN",
    "imaging_has_finding": "REVEALS",
    "finding_belongs_to_examination": "REVEALS",
}


@dataclass(frozen=True)
class ConversationDatasetLoadResult:
    """Validated splits + audit summary for the bilingual conversation dataset."""

    train: list[AnnotatedDocument]
    validation: list[AnnotatedDocument]
    test: list[AnnotatedDocument]
    excluded_records: list[dict[str, str]]
    audit: dict[str, Any]


def _map_assertion_and_temporality(span: dict[str, Any]) -> tuple[str, str]:
    raw_assert = str(span.get("assertion", "present")).strip().lower()
    raw_exp = str(span.get("experiencer", "patient")).strip().lower()
    raw_temp = str(span.get("temporality", "current")).strip().lower()

    if raw_temp in ("historical", "resolved"):
        canon_temp = "past"
    elif raw_temp == "planned":
        canon_temp = "planned"
    else:
        canon_temp = "current"

    if raw_exp == "family":
        canon_assert = "family"
    elif raw_assert == "absent":
        canon_assert = "absent"
    elif raw_assert in ("conditional", "hypothetical"):
        canon_assert = "hypothetical"
    elif raw_assert in ("possible", "conflicting"):
        canon_assert = "possible"
    elif canon_temp == "past" and raw_assert == "present":
        canon_assert = "historical"
    else:
        canon_assert = "present"

    return canon_assert, canon_temp


def _convert_conversation_record(rec: dict[str, Any]) -> tuple[AnnotatedDocument | None, str | None]:
    """Validate and convert one raw conversation JSON object into canonical AnnotatedDocument."""
    cid = str(rec.get("conversation_id") or "")
    text = str(rec.get("text") or "")
    if not cid or not text.strip():
        return None, "missing_id_or_empty_text"

    enc_id = str(rec.get("encounter_id") or cid)
    raw_ann = rec.get("annotation")
    ann: dict[str, Any] = raw_ann if isinstance(raw_ann, dict) else {}
    raw_pat = ann.get("patient")
    pat_obj: dict[str, Any] = raw_pat if isinstance(raw_pat, dict) else {}
    raw_enc = ann.get("encounter")
    enc_obj: dict[str, Any] = raw_enc if isinstance(raw_enc, dict) else {}
    patient_id = str(pat_obj.get("patient_id") or enc_id)
    doc_date = str(enc_obj.get("date") or "2026-11-26")[:10]
    raw_lang = str(rec.get("language") or "en").strip().lower()
    canon_lang = "en" if raw_lang == "en" else "fr"

    entities: list[EntitySpan] = []
    timex: list[TimexSpan] = []
    valid_eids: set[str] = set()
    occupied: list[tuple[int, int]] = []

    spans_val = rec.get("spans")
    raw_spans: list[dict[str, Any]] = spans_val if isinstance(spans_val, list) else []
    for sp in sorted(raw_spans, key=lambda x: (int(x.get("char_start", 0)), -int(x.get("char_end", 0)))):
        eid = str(sp.get("entity_id") or "")
        s = int(sp.get("char_start", -1))
        e = int(sp.get("char_end", -1))
        stxt = str(sp.get("text") or "")
        if s < 0 or e > len(text) or s >= e or text[s:e] != stxt:
            return None, f"span_offset_mismatch:{eid}"

        raw_etype = str(sp.get("entity_type") or "")
        mapped_type = CONVERSATION_ENTITY_MAP.get(raw_etype)
        if mapped_type is None:
            continue
        if any(max(s, os_) < min(e, oe_) for os_, oe_ in occupied):
            continue
        canon_assert, canon_temp = _map_assertion_and_temporality(sp)
        entities.append(
            EntitySpan(
                id=eid,
                type=mapped_type,
                start=s,
                end=e,
                text=stxt,
                assertion=canon_assert,
                temporality=canon_temp,
                normalized_code=str(sp.get("normalized_value")) if sp.get("normalized_value") is not None else None,
            )
        )
        valid_eids.add(eid)
        occupied.append((s, e))

        if mapped_type in ("DATE", "DURATION", "FREQUENCY"):
            timex.append(
                TimexSpan(
                    id=f"T_{eid}",
                    type=mapped_type,
                    start=s,
                    end=e,
                    text=stxt,
                    normalized=str(sp.get("normalized_value")) if sp.get("normalized_value") is not None else None,
                )
            )

    relations: list[RelationSpan] = []
    for rel in ann.get("relations", []):
        rid = str(rel.get("relation_id") or "")
        rtype = CONVERSATION_RELATION_MAP.get(str(rel.get("relation_type") or ""))
        hid = str(rel.get("head_entity_id") or "")
        tid = str(rel.get("tail_entity_id") or "")
        if rtype is None or hid not in valid_eids or tid not in valid_eids or hid == tid:
            continue
        relations.append(RelationSpan(id=rid, type=rtype, head=hid, tail=tid))

    doc = AnnotatedDocument(
        doc_id=cid,
        patient_id=patient_id,
        language=canon_lang,
        doc_type="consultation",
        document_date=doc_date,
        source="medical_conversation_dataset_en_fr",
        text=text,
        entities=entities,
        relations=relations,
        timex=timex,
        split=str(rec.get("split") or "train"),
        metadata={
            "encounter_id": enc_id,
            "paired_conversation_id": rec.get("paired_conversation_id"),
            "original_language": raw_lang,
            "noise_severity": str(rec.get("noise_severity") or "clean"),
            "noise_types": list(rec.get("noise_types") or []),
            "specialty": str(rec.get("specialty") or "general_medicine"),
            "n_contradictions": len(ann.get("contradictions") or []),
            "n_ocr_corrupted_spans": len([sp for sp in raw_spans if sp.get("ocr_corrupted")]),
        },
    )
    errs = doc.validate_spans()
    if errs:
        return None, f"schema_validation_failed:{errs[0]}"
    return doc, None


def load_conversation_dataset_from_zip(
    zip_path: Path | str = Path("medical_conversation_dataset_en_fr.zip"),
) -> ConversationDatasetLoadResult:
    """Validate archive integrity, check split leakage, and return canonical train/val/test splits."""
    zpath = Path(zip_path)
    if not zpath.exists():
        raise FileNotFoundError(f"Conversation dataset archive not found: {zpath}")

    train_docs: list[AnnotatedDocument] = []
    val_docs: list[AnnotatedDocument] = []
    test_docs: list[AnnotatedDocument] = []
    excluded: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    split_encounters: dict[str, set[str]] = defaultdict(set)
    lang_counts: Counter[str] = Counter()
    noise_counts: Counter[str] = Counter()

    with zipfile.ZipFile(zpath, "r") as zf:
        bad_member = zf.testzip()
        if bad_member is not None:
            raise zipfile.BadZipFile(f"Corrupt ZIP member: {bad_member}")
        raw_lines = zf.read("medical_conversation_dataset/annotations/jsonl/all_conversations.jsonl").decode("utf-8").splitlines()

    for line_no, line in enumerate(raw_lines):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            excluded.append({"id": f"line_{line_no}", "reason": f"json_error:{exc}"})
            continue
        cid = str(rec.get("conversation_id") or f"line_{line_no}")
        if cid in seen_ids:
            excluded.append({"id": cid, "reason": "duplicate_conversation_id"})
            continue
        seen_ids.add(cid)

        doc, err = _convert_conversation_record(rec)
        if doc is None:
            excluded.append({"id": cid, "reason": err or "unknown_error"})
            continue

        sp = doc.split or "train"
        split_encounters[sp].add(str(doc.metadata.get("encounter_id", cid)))
        lang_counts[str(doc.metadata.get("original_language", doc.language))] += 1
        noise_counts[str(doc.metadata.get("noise_severity", "clean"))] += 1

        if sp == "train":
            train_docs.append(doc)
        elif sp == "validation":
            val_docs.append(doc)
        elif sp == "test":
            test_docs.append(doc)
        else:
            excluded.append({"id": cid, "reason": f"unknown_split:{sp}"})

    leakage = {
        "train_val_encounter_overlap": len(split_encounters["train"] & split_encounters["validation"]),
        "train_test_encounter_overlap": len(split_encounters["train"] & split_encounters["test"]),
        "val_test_encounter_overlap": len(split_encounters["validation"] & split_encounters["test"]),
    }
    if any(v > 0 for v in leakage.values()):
        raise ValueError(f"Encounter leakage detected across splits: {leakage}")

    audit = {
        "archive_path": str(zpath),
        "total_loaded": len(train_docs) + len(val_docs) + len(test_docs),
        "train_n": len(train_docs),
        "validation_n": len(val_docs),
        "test_n": len(test_docs),
        "excluded_n": len(excluded),
        "leakage": leakage,
        "languages": dict(lang_counts),
        "noise_severities": dict(noise_counts),
    }
    return ConversationDatasetLoadResult(
        train=train_docs,
        validation=val_docs,
        test=test_docs,
        excluded_records=excluded,
        audit=audit,
    )
