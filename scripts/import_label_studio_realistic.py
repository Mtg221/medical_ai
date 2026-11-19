"""Import and strictly validate owner-annotated Label Studio JSON into AnnotatedDocument JSONL (+ IAA)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from medresearch_platform.data.schema import (
    ASSERTION_CLASSES,
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


def _parse_single_annotation(
    task: dict[str, Any],
    results: list[dict[str, Any]],
    *,
    doc_index: int,
    source: str,
) -> AnnotatedDocument:
    valid_types = set(ENTITY_TYPES)
    valid_assertions = set(ASSERTION_CLASSES)
    valid_relations = set(RELATION_TYPES)
    valid_timex = set(TIMEX_TYPES)

    data = task.get("data", {})
    text = str(data.get("text", ""))
    if not text:
        raise ValueError(f"Task {task.get('id')} has empty text")
    doc_id = str(data.get("doc_id") or f"ls_{task.get('id', doc_index + 1)}")
    language = str(data.get("language", "en"))
    doc_type = str(data.get("doc_type", "consultation"))
    document_date = str(data.get("document_date", "2025-01-01"))

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


def convert_label_studio_records(
    records: list[dict[str, Any]],
    *,
    source: str = "owner_realistic_annotated",
) -> list[AnnotatedDocument]:
    """Validate label schema and convert Label Studio export tasks to AnnotatedDocuments."""
    docs: list[AnnotatedDocument] = []
    for idx, task in enumerate(records):
        annotations = task.get("annotations", [])
        if not annotations:
            doc_id = str(task.get("data", {}).get("doc_id") or f"ls_{task.get('id', idx + 1)}")
            raise ValueError(f"Task {doc_id} has no annotations")
        docs.append(
            _parse_single_annotation(
                task,
                annotations[0].get("result", []),
                doc_index=idx,
                source=source,
            )
        )
    return docs


def compute_label_studio_iaa(
    records: list[dict[str, Any]],
    *,
    source: str = "owner_realistic_annotated",
) -> dict[str, Any]:
    """Compute Inter-Annotator Agreement across all tasks that carry >= 2 annotations."""
    pairs: list[tuple[AnnotatedDocument, AnnotatedDocument]] = []
    for idx, task in enumerate(records):
        annotations = task.get("annotations", [])
        if len(annotations) < 2:
            continue
        doc_a = _parse_single_annotation(task, annotations[0].get("result", []), doc_index=idx, source=source)
        doc_b = _parse_single_annotation(task, annotations[1].get("result", []), doc_index=idx, source=source)
        pairs.append((doc_a, doc_b))
    iaa = compute_document_pair_iaa(pairs)
    iaa["total_tasks"] = len(records)
    iaa["double_annotation_rate"] = round(len(pairs) / max(1, len(records)), 4)
    return iaa


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Label Studio JSON export for realistic evaluation.")
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
        "--source",
        type=str,
        default="owner_realistic_annotated",
        help="Provenance tag stored in AnnotatedDocument.source",
    )
    args = parser.parse_args()
    records = json.loads(args.input.read_text(encoding="utf-8"))
    docs = convert_label_studio_records(records, source=args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")
    print(
        f"Imported and validated {len(docs)} realistic documents "
        f"(entities={sum(len(d.entities) for d in docs)}, "
        f"relations={sum(len(d.relations) for d in docs)}, "
        f"timex={sum(len(d.timex) for d in docs)}) to {args.output}"
    )
    if args.iaa_out is not None:
        iaa = compute_label_studio_iaa(records, source=args.source)
        args.iaa_out.parent.mkdir(parents=True, exist_ok=True)
        args.iaa_out.write_text(json.dumps(iaa, indent=2), encoding="utf-8")
        print(
            f"Wrote IAA report ({iaa['double_annotated_docs']}/{iaa['total_tasks']} double-annotated, "
            f"entity_f1={iaa['entity_strict_f1']:.4f}, kappa={iaa['assertion_cohens_kappa']:.4f}) to {args.iaa_out}"
        )


if __name__ == "__main__":
    main()
