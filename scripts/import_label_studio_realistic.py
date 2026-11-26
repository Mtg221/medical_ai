"""Import, adjudicate, freeze, and strictly validate owner-annotated Label Studio JSON into AnnotatedDocument JSONL (+ IAA + SHA-256 manifest)."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from medresearch_platform.data.schema import (
    ASSERTION_CLASSES,
    DOC_TYPES,
    ENTITY_TYPES,
    RELATION_TYPES,
    TIMEX_TYPES,
    AnnotatedDocument,
    EntitySpan,
    RelationSpan,
    TimexSpan,
)
from medresearch_platform.eval_harness.iaa import compute_document_pair_iaa
from medresearch_platform.hybrid.stage4_temporal import normalize_timex

EXPECTED_TOTAL_DOCS = 80
EXPECTED_EN_DOCS = 40
EXPECTED_FR_DOCS = 40
MIN_DOUBLE_ANNOTATED_DOCS = 16  # >= 20% of 80
VALID_LANGUAGES = frozenset({"en", "fr"})


def _is_adjudicated_annotation(ann: dict[str, Any]) -> bool:
    """Return True if a Label Studio annotation explicitly marks itself as adjudicated gold."""
    if ann.get("ground_truth") is True or ann.get("adjudicated") is True:
        return True
    if str(ann.get("role", "")).strip().lower() == "adjudicated":
        return True
    meta = ann.get("meta")
    if isinstance(meta, dict) and (meta.get("adjudicated") is True or meta.get("ground_truth") is True):
        return True
    return False


def _canonical_doc_signature(
    doc: AnnotatedDocument,
) -> tuple[
    frozenset[tuple[str, int, int, str]],
    frozenset[tuple[str, tuple[str, int, int], tuple[str, int, int]]],
    frozenset[tuple[str, int, int]],
]:
    """Return a canonical comparable signature of entities, relations, and TIMEX spans."""
    ent_by_id = {e.id: (e.type, e.start, e.end) for e in doc.entities}
    ent_sig = frozenset((e.type, e.start, e.end, e.assertion) for e in doc.entities)
    rel_sig = frozenset(
        (r.type, ent_by_id[r.head], ent_by_id[r.tail])
        for r in doc.relations
        if r.head in ent_by_id and r.tail in ent_by_id
    )
    timex_sig = frozenset((t.type, t.start, t.end) for t in doc.timex)
    return ent_sig, rel_sig, timex_sig


def _parse_single_annotation(
    task: dict[str, Any],
    results: list[dict[str, Any]],
    *,
    doc_index: int,
    source: str,
    require_explicit_metadata: bool = False,
) -> AnnotatedDocument:
    valid_types = set(ENTITY_TYPES)
    valid_assertions = set(ASSERTION_CLASSES)
    valid_relations = set(RELATION_TYPES)
    valid_timex = set(TIMEX_TYPES)
    valid_doc_types = set(DOC_TYPES)

    data = task.get("data", {})
    text = str(data.get("text", ""))
    if not text or not text.strip():
        raise ValueError(f"Task {task.get('id')} has empty text")

    raw_doc_id = data.get("doc_id")
    if require_explicit_metadata and (not isinstance(raw_doc_id, str) or not raw_doc_id.strip()):
        raise ValueError(f"Task index {doc_index} (id={task.get('id')}) missing explicit 'doc_id' metadata")
    doc_id = str(raw_doc_id).strip() if raw_doc_id else f"ls_{task.get('id', doc_index + 1)}"

    raw_lang = data.get("language")
    if require_explicit_metadata:
        if not isinstance(raw_lang, str) or raw_lang.strip() not in VALID_LANGUAGES:
            raise ValueError(
                f"Task {doc_id} missing or invalid explicit 'language'={raw_lang!r}; expected one of {sorted(VALID_LANGUAGES)}"
            )
        language = raw_lang.strip()
    else:
        language = str(raw_lang or "en")

    raw_doc_type = data.get("doc_type")
    if require_explicit_metadata:
        if not isinstance(raw_doc_type, str) or raw_doc_type.strip() not in valid_doc_types:
            raise ValueError(
                f"Task {doc_id} missing or invalid explicit 'doc_type'={raw_doc_type!r}; "
                f"expected one of {sorted(valid_doc_types)}"
            )
        doc_type = raw_doc_type.strip()
    else:
        doc_type = str(raw_doc_type or "consultation")

    raw_date = data.get("document_date")
    if require_explicit_metadata:
        if not isinstance(raw_date, str) or not raw_date.strip():
            raise ValueError(f"Task {doc_id} missing explicit 'document_date' (YYYY-MM-DD)")
        try:
            date.fromisoformat(raw_date.strip())
        except ValueError as exc:
            raise ValueError(
                f"Task {doc_id} has invalid 'document_date'={raw_date!r}; expected valid ISO YYYY-MM-DD"
            ) from exc
        document_date = raw_date.strip()
    else:
        document_date = str(raw_date or "2025-01-01")

    assertion_by_region: dict[str, str] = {}
    for item in results:
        if item.get("type") == "choices" and item.get("from_name") == "assertion":
            choices = item.get("value", {}).get("choices", [])
            if choices:
                assertion_by_region[str(item.get("id", ""))] = str(choices[0])

    entities: list[EntitySpan] = []
    timex: list[TimexSpan] = []
    region_to_entity_id: dict[str, str] = {}
    seen_timex_keys: set[tuple[str, int, int]] = set()

    for item in results:
        if item.get("type") != "labels":
            continue
        from_name = str(item.get("from_name", "entities"))
        val = item.get("value", {})
        labels = val.get("labels", [])
        if not labels:
            raise ValueError(f"Task {doc_id} label region missing label name")
        label = str(labels[0])
        start = int(val["start"])
        end = int(val["end"])
        span_text = text[start:end]

        if from_name == "timex":
            if label not in valid_timex:
                raise ValueError(
                    f"Unknown timex label {label!r} in task {doc_id}; expected one of {sorted(valid_timex)}"
                )
            key = (label, start, end)
            if key not in seen_timex_keys:
                seen_timex_keys.add(key)
                timex.append(
                    TimexSpan(
                        id=f"t{len(timex) + 1}",
                        type=label,
                        start=start,
                        end=end,
                        text=span_text,
                        normalized=normalize_timex(span_text, label, document_date),
                    )
                )
            continue

        if label not in valid_types:
            raise ValueError(
                f"Unknown entity label {label!r} in task {doc_id}; expected one of {sorted(valid_types)}"
            )
        region_id = str(item.get("id", ""))
        assertion = assertion_by_region.get(region_id, "present")
        if assertion not in valid_assertions:
            raise ValueError(
                f"Unknown assertion {assertion!r} in task {doc_id}; expected one of {sorted(valid_assertions)}"
            )
        ent_id = f"e{len(entities) + 1}"
        if region_id:
            region_to_entity_id[region_id] = ent_id
        entities.append(
            EntitySpan(
                id=ent_id,
                type=label,
                start=start,
                end=end,
                text=span_text,
                assertion=assertion,
            )
        )
        if label in {"DATE", "DURATION", "FREQUENCY"}:
            key = (label, start, end)
            if key not in seen_timex_keys:
                seen_timex_keys.add(key)
                timex.append(
                    TimexSpan(
                        id=f"t{len(timex) + 1}",
                        type=label,
                        start=start,
                        end=end,
                        text=span_text,
                        normalized=normalize_timex(span_text, label, document_date),
                    )
                )

    relations: list[RelationSpan] = []
    for item in results:
        if item.get("type") != "relation":
            continue
        rel_labels = item.get("labels", [])
        if not rel_labels:
            raise ValueError(f"Task {doc_id} relation missing label name")
        rel_type = str(rel_labels[0])
        if rel_type not in valid_relations:
            raise ValueError(
                f"Unknown relation label {rel_type!r} in task {doc_id}; expected one of {sorted(valid_relations)}"
            )
        from_id = str(item.get("from_id", ""))
        to_id = str(item.get("to_id", ""))
        head_ent = region_to_entity_id.get(from_id)
        tail_ent = region_to_entity_id.get(to_id)
        if head_ent is None or tail_ent is None:
            raise ValueError(
                f"Task {doc_id} relation {rel_type!r} references unknown region IDs ({from_id!r} -> {to_id!r})"
            )
        relations.append(
            RelationSpan(
                id=f"r{len(relations) + 1}",
                type=rel_type,
                head=head_ent,
                tail=tail_ent,
            )
        )

    doc = AnnotatedDocument(
        doc_id=doc_id,
        patient_id=str(data.get("patient_id", f"pat_{doc_id}")),
        language=language,
        doc_type=doc_type,
        document_date=document_date,
        source=source,
        text=text,
        entities=entities,
        relations=relations,
        timex=sorted(timex, key=lambda t: t.start),
    )
    errs = doc.validate_spans()
    if errs:
        raise ValueError(f"Span validation failed for {doc_id}: {errs[:3]}")
    return doc


def _select_gold_annotation_for_task(
    task: dict[str, Any],
    *,
    doc_index: int,
    source: str,
    require_explicit_metadata: bool,
    require_adjudication: bool,
) -> tuple[AnnotatedDocument, bool]:
    """Return `(gold_doc, was_explicitly_adjudicated)` for a task, rejecting unadjudicated disagreements when required."""
    annotations = task.get("annotations", [])
    doc_id = str(task.get("data", {}).get("doc_id") or f"ls_{task.get('id', doc_index + 1)}")
    if not annotations:
        raise ValueError(f"Task {doc_id} has no annotations")

    adjudicated = [ann for ann in annotations if _is_adjudicated_annotation(ann)]
    if len(adjudicated) > 1:
        raise ValueError(
            f"Task {doc_id} has {len(adjudicated)} annotations marked as adjudicated/ground_truth; expected at most 1."
        )
    if len(adjudicated) == 1:
        return (
            _parse_single_annotation(
                task,
                adjudicated[0].get("result", []),
                doc_index=doc_index,
                source=source,
                require_explicit_metadata=require_explicit_metadata,
            ),
            True,
        )

    parsed_all = [
        _parse_single_annotation(
            task,
            ann.get("result", []),
            doc_index=doc_index,
            source=source,
            require_explicit_metadata=require_explicit_metadata,
        )
        for ann in annotations
    ]

    if len(parsed_all) >= 2 and require_adjudication:
        ref_sig = _canonical_doc_signature(parsed_all[0])
        for other_idx, other_doc in enumerate(parsed_all[1:], start=2):
            other_sig = _canonical_doc_signature(other_doc)
            if other_sig != ref_sig:
                diffs: list[str] = []
                if other_sig[0] != ref_sig[0]:
                    diffs.append("entities/assertions")
                if other_sig[1] != ref_sig[1]:
                    diffs.append("relations")
                if other_sig[2] != ref_sig[2]:
                    diffs.append("timex")
                raise ValueError(
                    f"Task {doc_id} has unadjudicated disagreement between annotator 1 and annotator {other_idx} "
                    f"on {', '.join(diffs)}. Never silently use annotator 1 as gold: adjudicate the task and mark "
                    "the adjudicated annotation with ground_truth=true (or adjudicated=true)."
                )
    return parsed_all[0], False


def validate_human_realistic_protocol(
    records: list[dict[str, Any]],
    docs: list[AnnotatedDocument],
    iaa: dict[str, Any],
) -> dict[str, Any]:
    """Enforce the full 80-note human protocol (40 EN / 40 FR, 6 doc types, >= 16 double-annotated, unique docs)."""
    errors: list[str] = []

    if len(docs) != EXPECTED_TOTAL_DOCS:
        errors.append(f"Expected exactly {EXPECTED_TOTAL_DOCS} documents, found {len(docs)}.")

    doc_ids = [d.doc_id for d in docs]
    dup_ids = [k for k, c in Counter(doc_ids).items() if c > 1]
    if dup_ids:
        errors.append(f"Duplicate doc_id values found: {sorted(dup_ids)[:5]}.")

    norm_texts = [re.sub(r"\s+", " ", d.text.strip().lower()) for d in docs]
    dup_text_count = len(norm_texts) - len(set(norm_texts))
    if dup_text_count > 0:
        errors.append(f"Found {dup_text_count} duplicate normalized document text(s); all 80 notes must be unique.")

    lang_counts = Counter(d.language for d in docs)
    if lang_counts.get("en", 0) != EXPECTED_EN_DOCS or lang_counts.get("fr", 0) != EXPECTED_FR_DOCS:
        errors.append(
            f"Expected {EXPECTED_EN_DOCS} EN and {EXPECTED_FR_DOCS} FR documents, found {dict(sorted(lang_counts.items()))}."
        )

    doc_type_counts = Counter(d.doc_type for d in docs)
    missing_doc_types = sorted(set(DOC_TYPES) - set(doc_type_counts.keys()))
    if missing_doc_types:
        errors.append(f"Missing required document types: {missing_doc_types} (must cover all 6 DOC_TYPES).")

    double_docs = int(iaa.get("double_annotated_docs", 0))
    if double_docs < MIN_DOUBLE_ANNOTATED_DOCS:
        errors.append(
            f"Expected >= {MIN_DOUBLE_ANNOTATED_DOCS} double-annotated documents (>= 20%), found {double_docs}."
        )

    if errors:
        raise ValueError("Human 80-note realistic protocol validation failed:\n- " + "\n- ".join(errors))

    return {
        "protocol_validated": True,
        "n_docs": len(docs),
        "language_counts": dict(sorted(lang_counts.items())),
        "doc_type_counts": dict(sorted(doc_type_counts.items())),
        "double_annotated_docs": double_docs,
        "double_annotation_rate": round(double_docs / max(1, len(docs)), 4),
    }


def convert_label_studio_records(
    records: list[dict[str, Any]],
    *,
    source: str = "owner_realistic_annotated",
    strict_protocol: bool = False,
    require_explicit_metadata: bool = False,
    require_adjudication: bool = False,
) -> list[AnnotatedDocument]:
    """Validate label schema and convert Label Studio export tasks to AnnotatedDocuments."""
    eff_explicit = bool(require_explicit_metadata or strict_protocol)
    eff_adjudication = bool(require_adjudication or strict_protocol)

    docs: list[AnnotatedDocument] = []
    for idx, task in enumerate(records):
        doc, _ = _select_gold_annotation_for_task(
            task,
            doc_index=idx,
            source=source,
            require_explicit_metadata=eff_explicit,
            require_adjudication=eff_adjudication,
        )
        docs.append(doc)

    if strict_protocol:
        iaa = compute_label_studio_iaa(
            records,
            source=source,
            require_explicit_metadata=eff_explicit,
        )
        validate_human_realistic_protocol(records, docs, iaa)

    return docs


def compute_label_studio_iaa(
    records: list[dict[str, Any]],
    *,
    source: str = "owner_realistic_annotated",
    require_explicit_metadata: bool = False,
) -> dict[str, Any]:
    """Compute Inter-Annotator Agreement across all tasks that carry >= 2 independent (non-adjudicated) annotations."""
    pairs: list[tuple[AnnotatedDocument, AnnotatedDocument]] = []
    for idx, task in enumerate(records):
        annotations = task.get("annotations", [])
        independent = [ann for ann in annotations if not _is_adjudicated_annotation(ann)]
        if len(independent) < 2:
            continue
        doc_a = _parse_single_annotation(
            task,
            independent[0].get("result", []),
            doc_index=idx,
            source=source,
            require_explicit_metadata=require_explicit_metadata,
        )
        doc_b = _parse_single_annotation(
            task,
            independent[1].get("result", []),
            doc_index=idx,
            source=source,
            require_explicit_metadata=require_explicit_metadata,
        )
        pairs.append((doc_a, doc_b))
    iaa = compute_document_pair_iaa(pairs)
    iaa["total_tasks"] = len(records)
    iaa["double_annotation_rate"] = round(len(pairs) / max(1, len(records)), 4)
    return iaa


def build_freeze_manifest(
    *,
    input_path: Path,
    output_path: Path,
    docs: list[AnnotatedDocument],
    iaa: dict[str, Any],
    source: str,
    dataset_version: str,
    protocol_validated: bool,
    adjudicated_docs: int,
) -> dict[str, Any]:
    """Build an immutable provenance + SHA-256 freeze manifest for the imported human realistic set."""
    lang_counts = dict(sorted(Counter(d.language for d in docs).items()))
    doc_type_counts = dict(sorted(Counter(d.doc_type for d in docs).items()))
    return {
        "dataset_version": dataset_version,
        "frozen_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": source,
        "protocol_validated": bool(protocol_validated),
        "input_path": str(input_path),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "output_path": str(output_path),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "n_docs": len(docs),
        "language_counts": lang_counts,
        "doc_type_counts": doc_type_counts,
        "double_annotated_docs": int(iaa.get("double_annotated_docs", 0)),
        "double_annotation_rate": float(iaa.get("double_annotation_rate", 0.0)),
        "adjudicated_docs": int(adjudicated_docs),
        "gold_counts": {
            "entities": sum(len(d.entities) for d in docs),
            "relations": sum(len(d.relations) for d in docs),
            "timex": sum(len(d.timex) for d in docs),
        },
        "iaa_summary": iaa,
        "freeze_policy": (
            "FROZEN BLIND EVALUATION SET: Do not inspect errors to tune models, prompts, thresholds, "
            "or features. Evaluate once with scripts/eval_realistic.py --owner-written (post_hoc=false)."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import, adjudicate, and freeze Label Studio JSON export for human 80-note realistic evaluation."
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to Label Studio JSON export")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/realistic/eval.jsonl"),
        help="Destination validated JSONL path",
    )
    parser.add_argument(
        "--iaa-out",
        type=Path,
        default=None,
        help="Optional JSON path to write double-annotation IAA summary",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="Path to write SHA-256 freeze manifest (defaults to <output>.manifest.json)",
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        default="human-realistic-v1.0",
        help="Version tag recorded in the freeze manifest",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="owner_realistic_annotated",
        help="Provenance tag stored in AnnotatedDocument.source",
    )
    parser.add_argument(
        "--allow-partial-dev",
        action="store_true",
        help="Relax 80-note / 40 EN / 40 FR / 6 doc-type / >=16 double-annotation quotas for dev smoke tests ONLY",
    )
    args = parser.parse_args()
    strict_protocol = not args.allow_partial_dev

    records = json.loads(args.input.read_text(encoding="utf-8"))
    adjudicated_docs = 0
    docs: list[AnnotatedDocument] = []
    for idx, task in enumerate(records):
        doc, was_adj = _select_gold_annotation_for_task(
            task,
            doc_index=idx,
            source=args.source,
            require_explicit_metadata=strict_protocol,
            require_adjudication=strict_protocol,
        )
        if was_adj:
            adjudicated_docs += 1
        docs.append(doc)

    iaa = compute_label_studio_iaa(
        records,
        source=args.source,
        require_explicit_metadata=strict_protocol,
    )
    if strict_protocol:
        validate_human_realistic_protocol(records, docs, iaa)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")

    manifest_path = args.manifest_out or Path(f"{args.output}.manifest.json")
    manifest = build_freeze_manifest(
        input_path=args.input,
        output_path=args.output,
        docs=docs,
        iaa=iaa,
        source=args.source,
        dataset_version=args.dataset_version,
        protocol_validated=strict_protocol,
        adjudicated_docs=adjudicated_docs,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"Imported and validated {len(docs)} realistic documents "
        f"(protocol_validated={strict_protocol}, adjudicated={adjudicated_docs}, "
        f"entities={sum(len(d.entities) for d in docs)}, "
        f"relations={sum(len(d.relations) for d in docs)}, "
        f"timex={sum(len(d.timex) for d in docs)}) to {args.output}"
    )
    print(f"Wrote SHA-256 freeze manifest to {manifest_path} (sha256={manifest['output_sha256'][:16]})")
    if args.iaa_out is not None:
        args.iaa_out.parent.mkdir(parents=True, exist_ok=True)
        args.iaa_out.write_text(json.dumps(iaa, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f"Wrote IAA report ({iaa['double_annotated_docs']}/{iaa['total_tasks']} double-annotated, "
            f"entity_f1={iaa['entity_strict_f1']:.4f}, kappa={iaa['assertion_cohens_kappa']:.4f}) to {args.iaa_out}"
        )


if __name__ == "__main__":
    main()
