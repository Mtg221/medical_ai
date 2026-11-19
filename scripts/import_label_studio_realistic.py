"""Import and strictly validate owner-annotated Label Studio JSON into AnnotatedDocument JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from medresearch_platform.data.schema import (
    ASSERTION_CLASSES,
    ENTITY_TYPES,
    AnnotatedDocument,
    EntitySpan,
)


def convert_label_studio_records(
    records: list[dict[str, Any]],
    *,
    source: str = "owner_realistic_annotated",
) -> list[AnnotatedDocument]:
    """Validate label schema and convert Label Studio export tasks to AnnotatedDocuments."""
    valid_types = set(ENTITY_TYPES)
    valid_assertions = set(ASSERTION_CLASSES)
    docs: list[AnnotatedDocument] = []

    for task in records:
        data = task.get("data", {})
        text = str(data.get("text", ""))
        if not text:
            raise ValueError(f"Task {task.get('id')} has empty text")
        doc_id = str(data.get("doc_id") or f"ls_{task.get('id', len(docs) + 1)}")
        language = str(data.get("language", "en"))
        doc_type = str(data.get("doc_type", "consultation"))
        document_date = str(data.get("document_date", "2025-01-01"))

        annotations = task.get("annotations", [])
        if not annotations:
            raise ValueError(f"Task {doc_id} has no annotations")
        results = annotations[0].get("result", [])

        assertion_by_region: dict[str, str] = {}
        for item in results:
            if item.get("type") == "choices" and item.get("from_name") == "assertion":
                choices = item.get("value", {}).get("choices", [])
                if choices:
                    assertion_by_region[str(item.get("id", ""))] = str(choices[0])

        entities: list[EntitySpan] = []
        for idx, item in enumerate(results):
            if item.get("type") != "labels":
                continue
            val = item.get("value", {})
            labels = val.get("labels", [])
            if not labels:
                raise ValueError(f"Task {doc_id} label region missing label name")
            label = str(labels[0])
            if label not in valid_types:
                raise ValueError(
                    f"Unknown entity label {label!r} in task {doc_id}; expected one of {sorted(valid_types)}"
                )
            start = int(val["start"])
            end = int(val["end"])
            span_text = text[start:end]
            assertion = assertion_by_region.get(str(item.get("id", "")), "present")
            if assertion not in valid_assertions:
                raise ValueError(
                    f"Unknown assertion {assertion!r} in task {doc_id}; expected one of {sorted(valid_assertions)}"
                )
            entities.append(
                EntitySpan(
                    id=f"e{idx + 1}",
                    type=label,
                    start=start,
                    end=end,
                    text=span_text,
                    assertion=assertion,
                )
            )

        docs.append(
            AnnotatedDocument(
                doc_id=doc_id,
                patient_id=str(data.get("patient_id", f"pat_{doc_id}")),
                language=language,
                doc_type=doc_type,
                document_date=document_date,
                source=source,
                text=text,
                entities=entities,
            )
        )
    return docs


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
    print(f"Imported and validated {len(docs)} realistic documents to {args.output}")


if __name__ == "__main__":
    main()
