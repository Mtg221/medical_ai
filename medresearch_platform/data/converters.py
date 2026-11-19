"""Format converters (BRAT standoff, Label Studio JSON) and explicit external label mappings."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from typing import Any

from medresearch_platform.data.schema import (
    AnnotatedDocument,
    EntitySpan,
    RelationSpan,
    TimexSpan,
)
from medresearch_platform.hybrid.stage4_temporal import normalize_timex

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
        "BODYPART": "ANATOMY",
        "RML": "LAB_VALUE",
        "DATE": "DATE",
        "DURATION": "DURATION",
        "FREQUENCY": "FREQUENCY",
    },
    "maccrobat": {
        "Disease_disorder": "DIAGNOSIS",
        "Sign_symptom": "SYMPTOM",
        "Medication": "MEDICATION",
        "Dosage": "DOSAGE",
        "Frequency": "FREQUENCY",
        "Administration": "ROUTE",
        "Duration": "DURATION",
        "Diagnostic_procedure": "PROCEDURE",
        "Therapeutic_procedure": "PROCEDURE",
        "Lab_value": "LAB_VALUE",
        "Biological_structure": "ANATOMY",
        "Date": "DATE",
    },
    "pxcorpus": {
        "drug": "MEDICATION",
        "inn": "MEDICATION",
        "d_dos_val": "DOSAGE",
        "d_dos_up": "DOSAGE",
        "dos_val": "DOSAGE",
        "dos_uf": "DOSAGE",
        "dur_val": "DURATION",
        "dur_ut": "DURATION",
        "roa": "ROUTE",
        "rhythm_tdte": "FREQUENCY",
        "rhythm_perday": "FREQUENCY",
        "rhythm_hour": "FREQUENCY",
        "rhythm_rec_val": "FREQUENCY",
        "rhythm_rec_ut": "FREQUENCY",
        "freq_val": "FREQUENCY",
        "freq_ut": "FREQUENCY",
        "freq_days": "FREQUENCY",
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


def _dedupe_non_overlapping_entities(candidates: list[EntitySpan]) -> list[EntitySpan]:
    """Keep longest non-overlapping entity spans sorted by start offset."""
    ordered = sorted(candidates, key=lambda e: (e.start, -(e.end - e.start), e.type))
    kept: list[EntitySpan] = []
    for ent in ordered:
        if any(max(ent.start, prev.start) < min(ent.end, prev.end) for prev in kept):
            continue
        kept.append(ent)
    return sorted(kept, key=lambda e: e.start)


def from_maccrobat_brat(
    text: str,
    ann_text: str,
    *,
    doc_id: str,
    document_date: str = "2020-01-01",
    license_name: str = "CC-BY-4.0",
) -> AnnotatedDocument:
    """Convert a MACCROBAT BRAT standoff (.txt/.ann) case report to canonical AnnotatedDocument."""
    label_map = EXTERNAL_LABEL_MAPS["maccrobat"]
    raw_by_tid: dict[str, EntitySpan] = {}
    relations_raw: list[tuple[str, str, str]] = []

    for raw_line in ann_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        tag = parts[0]
        if tag.startswith("T") and len(parts) >= 3:
            head_parts = parts[1].split()
            raw_type = head_parts[0]
            canonical = label_map.get(raw_type)
            if canonical is None:
                continue
            # BRAT discontinuous spans look like "Type 10 15;20 25" -> take first contiguous span
            span_spec = parts[1][len(raw_type) :].strip().split(";")[0].strip()
            coords = span_spec.split()
            if len(coords) < 2:
                continue
            start, end = int(coords[0]), int(coords[1])
            if start < 0 or end > len(text) or start >= end:
                continue
            raw_by_tid[tag] = EntitySpan(
                id=tag,
                type=canonical,
                start=start,
                end=end,
                text=text[start:end],
                assertion="present",
                temporality="current",
            )
        elif tag.startswith("R") and len(parts) >= 2:
            rel_parts = parts[1].split()
            if len(rel_parts) == 3 and rel_parts[0] == "CAUSE":
                arg1 = rel_parts[1].split(":", 1)[-1]
                arg2 = rel_parts[2].split(":", 1)[-1]
                relations_raw.append(("CAUSES", arg1, arg2))

    kept = _dedupe_non_overlapping_entities(list(raw_by_tid.values()))
    old_to_new: dict[str, str] = {}
    entities: list[EntitySpan] = []
    timex: list[TimexSpan] = []
    for idx, ent in enumerate(kept, start=1):
        new_id = f"e{idx}"
        old_to_new[ent.id] = new_id
        entities.append(
            EntitySpan(
                id=new_id,
                type=ent.type,
                start=ent.start,
                end=ent.end,
                text=ent.text,
                assertion=ent.assertion,
                temporality=ent.temporality,
            )
        )
        if ent.type in {"DATE", "DURATION", "FREQUENCY"}:
            timex.append(
                TimexSpan(
                    id=f"t{len(timex) + 1}",
                    type=ent.type,
                    start=ent.start,
                    end=ent.end,
                    text=ent.text,
                    normalized=normalize_timex(ent.text, ent.type, document_date),
                )
            )

    relations: list[RelationSpan] = []
    for rtype, h_old, t_old in relations_raw:
        h_new = old_to_new.get(h_old)
        t_new = old_to_new.get(t_old)
        if h_new and t_new and h_new != t_new:
            relations.append(RelationSpan(id=f"r{len(relations) + 1}", type=rtype, head=h_new, tail=t_new))

    return AnnotatedDocument(
        doc_id=doc_id,
        patient_id=f"pat_{doc_id}",
        language="en",
        doc_type="discharge_summary",
        document_date=document_date,
        source="maccrobat2020",
        text=text,
        entities=entities,
        relations=relations,
        timex=timex,
        metadata={"corpus": "MACCROBAT2020", "license": license_name, "doi": "10.6084/m9.figshare.9764942"},
    )


def from_pxcorpus_row(
    row: dict[str, Any],
    tag_names: list[str],
    *,
    split: str = "train",
    document_date: str = "2022-04-25",
) -> AnnotatedDocument | None:
    """Convert a DrBenchmark/PxCorpus French prescription row into a validated AnnotatedDocument."""
    tokens = [str(t) for t in row.get("tokens", []) if str(t)]
    tag_ids = list(row.get("ner_tags", []))
    if not tokens or len(tokens) != len(tag_ids):
        return None

    # Reconstruct canonical single-space text with exact token char offsets
    text_parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for tok in tokens:
        if text_parts:
            text_parts.append(" ")
            cursor += 1
        start = cursor
        text_parts.append(tok)
        cursor += len(tok)
        offsets.append((start, cursor))
    text = "".join(text_parts)

    label_map = EXTERNAL_LABEL_MAPS["pxcorpus"]
    raw_spans: list[tuple[str, int, int]] = []
    cur_type: str | None = None
    cur_start = 0
    cur_end = 0

    for (t_start, t_end), tid in zip(offsets, tag_ids, strict=True):
        raw_tag = tag_names[int(tid)] if 0 <= int(tid) < len(tag_names) else "O"
        if raw_tag == "O" or "-" not in raw_tag:
            if cur_type is not None:
                raw_spans.append((cur_type, cur_start, cur_end))
                cur_type = None
            continue
        prefix, sub = raw_tag.split("-", 1)
        mapped = label_map.get(sub)
        if mapped is None:
            if cur_type is not None:
                raw_spans.append((cur_type, cur_start, cur_end))
                cur_type = None
            continue
        # Merge adjacent tokens sharing the same canonical type (e.g., d_dos_val + d_dos_up -> DOSAGE)
        if cur_type == mapped and (prefix == "I" or t_start <= cur_end + 1):
            cur_end = t_end
        else:
            if cur_type is not None:
                raw_spans.append((cur_type, cur_start, cur_end))
            cur_type = mapped
            cur_start = t_start
            cur_end = t_end
    if cur_type is not None:
        raw_spans.append((cur_type, cur_start, cur_end))

    entities: list[EntitySpan] = []
    timex: list[TimexSpan] = []
    for idx, (etype, start, end) in enumerate(raw_spans, start=1):
        span_txt = text[start:end]
        entities.append(
            EntitySpan(
                id=f"e{idx}",
                type=etype,
                start=start,
                end=end,
                text=span_txt,
                assertion="present",
                temporality="current",
            )
        )
        if etype in {"DURATION", "FREQUENCY"}:
            timex.append(
                TimexSpan(
                    id=f"t{len(timex) + 1}",
                    type=etype,
                    start=start,
                    end=end,
                    text=span_txt,
                    normalized=normalize_timex(span_txt, etype, document_date),
                )
            )

    row_id = str(row.get("id", "0"))
    doc_id = f"pxcorpus_{split}_{row_id}"
    return AnnotatedDocument(
        doc_id=doc_id,
        patient_id=f"pat_{doc_id}",
        language="fr",
        doc_type="prescription",
        document_date=document_date,
        source="pxcorpus",
        text=text,
        entities=entities,
        timex=timex,
        split=split,
        metadata={"corpus": "PxCorpus", "license": "CC-BY-4.0", "doi": "10.5281/zenodo.6482586"},
    )


def from_e3c_xmi(
    xmi_text: str,
    *,
    doc_id: str,
    language: str,
) -> AnnotatedDocument | None:
    """Convert an E3C Layer 1 UIMA XMI clinical case report to canonical AnnotatedDocument."""
    root = ET.fromstring(xmi_text)
    sofa = ""
    doc_date = "2020-01-01"
    doc_license = "CC-BY-NC-4.0"
    doc_source = "E3C-Corpus"
    doc_doi = ""

    events: list[tuple[int, int, str, str]] = []
    raw_entities: list[EntitySpan] = []
    timex_list: list[TimexSpan] = []

    for elem in root:
        local = elem.tag.split("}", 1)[-1]
        if local == "Sofa":
            sofa = elem.attrib.get("sofaString", "")
        elif local == "METADATA":
            raw_time = elem.attrib.get("docTime", "").strip()
            if len(raw_time) >= 10 and raw_time[4] == "-" and raw_time[7] == "-":
                doc_date = raw_time[:10]
            if elem.attrib.get("docLicense"):
                doc_license = elem.attrib["docLicense"].strip()
            if elem.attrib.get("docSource"):
                doc_source = elem.attrib["docSource"].strip()
            if elem.attrib.get("docDOI"):
                doc_doi = elem.attrib["docDOI"].strip()
        elif local == "EVENT":
            b = int(elem.attrib.get("begin", 0))
            e = int(elem.attrib.get("end", 0))
            if e > b:
                events.append(
                    (
                        b,
                        e,
                        elem.attrib.get("polarity", "POS").upper(),
                        elem.attrib.get("contextualModality", "ACTUAL").upper(),
                    )
                )

    if not sofa:
        return None

    def _assertion_for_span(start: int, end: int) -> str:
        for eb, ee, pol, mod in events:
            if max(start, eb) < min(end, ee):
                if pol == "NEG":
                    return "absent"
                if mod == "HYPOTHETICAL":
                    return "hypothetical"
                if mod == "HEDGED":
                    return "possible"
        return "present"

    timex_class_map = {"DATE": "DATE", "DURATION": "DURATION", "SET": "FREQUENCY", "TIME": "DATE"}
    for elem in root:
        local = elem.tag.split("}", 1)[-1]
        b = int(elem.attrib.get("begin", 0))
        e = int(elem.attrib.get("end", 0))
        if b < 0 or e > len(sofa) or e <= b:
            continue
        span_txt = sofa[b:e]
        if local == "CLINENTITY":
            raw_entities.append(
                EntitySpan(
                    id=f"c_{b}_{e}",
                    type="DIAGNOSIS",
                    start=b,
                    end=e,
                    text=span_txt,
                    assertion=_assertion_for_span(b, e),
                    normalized_code=elem.attrib.get("entityID") or None,
                    normalized_system="UMLS" if elem.attrib.get("entityID") else None,
                )
            )
        elif local == "BODYPART":
            raw_entities.append(
                EntitySpan(
                    id=f"b_{b}_{e}",
                    type="ANATOMY",
                    start=b,
                    end=e,
                    text=span_txt,
                    assertion="present",
                )
            )
        elif local == "RML":
            raw_entities.append(
                EntitySpan(
                    id=f"r_{b}_{e}",
                    type="LAB_VALUE",
                    start=b,
                    end=e,
                    text=span_txt,
                    assertion="present",
                )
            )
        elif local == "TIMEX3":
            tclass = timex_class_map.get(elem.attrib.get("timex3Class", "").upper())
            if tclass is None:
                continue
            val = elem.attrib.get("value") or normalize_timex(span_txt, tclass, doc_date)
            timex_list.append(
                TimexSpan(
                    id=f"t{len(timex_list) + 1}",
                    type=tclass,
                    start=b,
                    end=e,
                    text=span_txt,
                    normalized=val,
                )
            )
            raw_entities.append(
                EntitySpan(
                    id=f"tx_{b}_{e}",
                    type=tclass,
                    start=b,
                    end=e,
                    text=span_txt,
                    assertion="present",
                )
            )

    kept = _dedupe_non_overlapping_entities(raw_entities)
    entities = [
        EntitySpan(
            id=f"e{idx}",
            type=ent.type,
            start=ent.start,
            end=ent.end,
            text=ent.text,
            assertion=ent.assertion,
            temporality=ent.temporality,
            normalized_code=ent.normalized_code,
            normalized_system=ent.normalized_system,
        )
        for idx, ent in enumerate(kept, start=1)
    ]

    return AnnotatedDocument(
        doc_id=doc_id,
        patient_id=f"pat_{doc_id}",
        language=language,
        doc_type="discharge_summary",
        document_date=doc_date,
        source="e3c_layer1",
        text=sofa,
        entities=entities,
        timex=sorted(timex_list, key=lambda t: t.start),
        metadata={
            "corpus": "E3C-Corpus-Layer1",
            "corpus_license": "CC-BY-NC-4.0",
            "license": doc_license,
            "source_journal": doc_source,
            "doi": doc_doi,
        },
    )
