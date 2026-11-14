"""Format converters (BRAT standoff, Label Studio JSON) and explicit external label mappings."""

from __future__ import annotations

from collections import Counter
from typing import Any

from medresearch_platform.data.schema import (
    AnnotatedDocument,
    EntitySpan,
    RelationSpan,
)

EXTERNAL_LABEL_MAPS: dict[str, dict[str, str]] = {
    "quaero": {
        "DISO": "DIAGNOSIS",
        "CHEM": "MEDICATION",
        "PROC": "PROCEDURE",
        "ANAT": "ANATOMY",
    },
    "e3c": {
        "CLINENTITY": "DIAGNOSIS",
        "DISORDER": "DIAGNOSIS",
        "SYMPTOM": "SYMPTOM",
        "MEDICATION": "MEDICATION",
        "PROCEDURE": "PROCEDURE",
        "ANATOMY": "ANATOMY",
        "DATE": "DATE",
        "DURATION": "DURATION",
    },
    "i2b2_2010": {
        "problem": "DIAGNOSIS",
        "treatment": "MEDICATION",
        "test": "LAB_TEST",
    },
    "bc5cdr": {
        "Disease": "DIAGNOSIS",
        "Chemical": "MEDICATION",
    },
}


def map_external_label(source_corpus: str, raw_label: str) -> tuple[str | None, bool]:
    """Map an external corpus label to the canonical schema, returning (mapped_label, is_mapped)."""
    mapping = EXTERNAL_LABEL_MAPS.get(source_corpus.lower(), {})
    mapped = mapping.get(raw_label)
    return mapped, mapped is not None


def summarize_label_mapping(source_corpus: str, raw_labels: list[str]) -> dict[str, Any]:
    """Map a batch of external labels and report exact counts of mapped vs unmapped labels."""
    mapped_counts: Counter[str] = Counter()
    unmapped_counts: Counter[str] = Counter()
    for lbl in raw_labels:
        target, ok = map_external_label(source_corpus, lbl)
        if ok and target is not None:
            mapped_counts[f"{lbl}->{target}"] += 1
        else:
            unmapped_counts[lbl] += 1
    return {
        "source_corpus": source_corpus,
        "total": len(raw_labels),
        "mapped": dict(mapped_counts),
        "unmapped": dict(unmapped_counts),
    }


def to_brat_standoff(doc: AnnotatedDocument) -> tuple[str, str]:
    """Export an AnnotatedDocument to BRAT standoff (.txt, .ann) strings."""
    lines: list[str] = []
    brat_id_map: dict[str, str] = {}
    attr_idx = 1

    for idx, ent in enumerate(doc.entities, start=1):
        tid = f"T{idx}"
        brat_id_map[ent.id] = tid
        clean_text = ent.text.replace("\n", " ")
        lines.append(f"{tid}\t{ent.type} {ent.start} {ent.end}\t{clean_text}")
        if ent.assertion != "present":
            lines.append(f"A{attr_idx}\tAssertion {tid} {ent.assertion}")
            attr_idx += 1
        if ent.temporality != "current":
            lines.append(f"A{attr_idx}\tTemporality {tid} {ent.temporality}")
            attr_idx += 1

    for idx, rel in enumerate(doc.relations, start=1):
        head_t = brat_id_map.get(rel.head)
        tail_t = brat_id_map.get(rel.tail)
        if head_t and tail_t:
            lines.append(f"R{idx}\t{rel.type} Arg1:{head_t} Arg2:{tail_t}")

    return doc.text, "\n".join(lines) + ("\n" if lines else "")


def from_brat_standoff(
    text: str,
    ann_text: str,
    *,
    doc_id: str,
    patient_id: str = "PAT-BRAT",
    language: str = "en",
    doc_type: str = "discharge_summary",
    document_date: str = "2026-01-01",
    source: str = "brat_import",
) -> AnnotatedDocument:
    """Import BRAT standoff (.txt + .ann) into an AnnotatedDocument."""
    entities_by_tid: dict[str, dict[str, Any]] = {}
    relations_raw: list[tuple[str, str, str, str]] = []
    assertions: dict[str, str] = {}
    temporalities: dict[str, str] = {}

    for raw_line in ann_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        tag = parts[0]
        if tag.startswith("T") and len(parts) >= 3:
            meta = parts[1].split()
            etype = meta[0]
            start = int(meta[1])
            end = int(meta[-1].split(";")[0])
            entities_by_tid[tag] = {
                "id": tag,
                "type": etype,
                "start": start,
                "end": end,
                "text": text[start:end],
            }
        elif tag.startswith("A") and len(parts) >= 2:
            attr_parts = parts[1].split()
            if len(attr_parts) == 3:
                attr_name, tid, attr_val = attr_parts
                if attr_name == "Assertion":
                    assertions[tid] = attr_val
                elif attr_name == "Temporality":
                    temporalities[tid] = attr_val
        elif tag.startswith("R") and len(parts) >= 2:
            rel_parts = parts[1].split()
            if len(rel_parts) == 3:
                rtype = rel_parts[0]
                arg1 = rel_parts[1].split(":", 1)[1]
                arg2 = rel_parts[2].split(":", 1)[1]
                relations_raw.append((tag, rtype, arg1, arg2))

    entities = [
        EntitySpan(
            id=tid,
            type=data["type"],
            start=data["start"],
            end=data["end"],
            text=data["text"],
            assertion=assertions.get(tid, "present"),
            temporality=temporalities.get(tid, "current"),
        )
        for tid, data in entities_by_tid.items()
    ]
    relations = [
        RelationSpan(id=rid, type=rtype, head=h, tail=t)
        for rid, rtype, h, t in relations_raw
        if h in entities_by_tid and t in entities_by_tid
    ]
    return AnnotatedDocument(
        doc_id=doc_id,
        patient_id=patient_id,
        language=language,
        doc_type=doc_type,
        document_date=document_date,
        source=source,
        text=text,
        entities=entities,
        relations=relations,
    )


def to_label_studio(doc: AnnotatedDocument) -> dict[str, Any]:
    """Export an AnnotatedDocument to Label Studio JSON task format."""
    results: list[dict[str, Any]] = []
    for ent in doc.entities:
        results.append(
            {
                "id": ent.id,
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "value": {
                    "start": ent.start,
                    "end": ent.end,
                    "text": ent.text,
                    "labels": [ent.type],
                },
                "meta": {"assertion": ent.assertion, "temporality": ent.temporality},
            }
        )
    return {
        "data": {
            "text": doc.text,
            "doc_id": doc.doc_id,
            "patient_id": doc.patient_id,
            "language": doc.language,
            "doc_type": doc.doc_type,
            "document_date": doc.document_date,
        },
        "annotations": [{"result": results}],
    }


def from_label_studio(task: dict[str, Any]) -> AnnotatedDocument:
    """Import a Label Studio JSON task into an AnnotatedDocument."""
    data = task.get("data", {})
    text = str(data.get("text", ""))
    annotations = task.get("annotations", [])
    results = annotations[0].get("result", []) if annotations else []
    entities: list[EntitySpan] = []

    for idx, item in enumerate(results, start=1):
        if item.get("type") != "labels":
            continue
        val = item.get("value", {})
        labels = val.get("labels", [])
        if not labels:
            continue
        start = int(val["start"])
        end = int(val["end"])
        meta = item.get("meta", {})
        entities.append(
            EntitySpan(
                id=str(item.get("id", f"e{idx}")),
                type=str(labels[0]),
                start=start,
                end=end,
                text=text[start:end],
                assertion=str(meta.get("assertion", "present")),
                temporality=str(meta.get("temporality", "current")),
            )
        )

    return AnnotatedDocument(
        doc_id=str(data.get("doc_id", "ls_doc")),
        patient_id=str(data.get("patient_id", "PAT-LS")),
        language=str(data.get("language", "en")),
        doc_type=str(data.get("doc_type", "discharge_summary")),
        document_date=str(data.get("document_date", "2026-01-01")),
        source="label_studio_import",
        text=text,
        entities=entities,
    )
