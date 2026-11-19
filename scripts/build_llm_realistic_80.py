"""Build, strictly validate, and export the 80 LLM-authored EN+FR realistic clinical notes in Label Studio JSON + AnnotatedDocument JSONL."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from import_label_studio_realistic import convert_label_studio_records
from realistic_llm_en_notes import EN_NOTES
from realistic_llm_fr_notes import FR_NOTES

from medresearch_platform.data.schema import ASSERTION_CLASSES, DOC_TYPES, ENTITY_TYPES


def _find_occurrence(text: str, sub: str, occ: int) -> tuple[int, int]:
    """Find the `occ`-th (0-based) boundary-clean occurrence of `sub` in `text`."""
    # Prefer token-boundary matches when sub starts/ends with alphanumeric, fallback to exact substring search.
    starts_alnum = sub[0].isalnum()
    ends_alnum = sub[-1].isalnum()
    pat = re.escape(sub)
    if starts_alnum:
        if sub[0].isdigit():
            pat = r"(?<![A-Za-zÀ-ÖØ-öø-ÿ0-9])(?<![0-9][.,])" + pat
        else:
            pat = r"(?<![A-Za-zÀ-ÖØ-öø-ÿ0-9])" + pat
    if ends_alnum:
        if sub[-1].isdigit():
            pat = pat + r"(?![A-Za-zÀ-ÖØ-öø-ÿ0-9])(?![.,][0-9])"
        else:
            pat = pat + r"(?![A-Za-zÀ-ÖØ-öø-ÿ0-9])"
    matches = list(re.finditer(pat, text))
    if len(matches) <= occ:
        # Fallback to plain substring occurrences (for units like `%` or abbreviations attached to punctuation)
        plain: list[tuple[int, int]] = []
        pos = 0
        while True:
            idx = text.find(sub, pos)
            if idx == -1:
                break
            plain.append((idx, idx + len(sub)))
            pos = idx + 1
        if len(plain) <= occ:
            raise ValueError(f"Substring {sub!r} (occ={occ}) not found in text: {text[:120]!r}")
        return plain[occ]
    m = matches[occ]
    return m.start(), m.end()


def build_label_studio_export(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid_types = set(ENTITY_TYPES)
    valid_assertions = set(ASSERTION_CLASSES)
    valid_doc_types = set(DOC_TYPES)

    tasks: list[dict[str, Any]] = []
    for task_idx, note in enumerate(notes, start=1):
        doc_id = str(note["doc_id"])
        text = str(note["text"])
        doc_type = str(note["doc_type"])
        if doc_type not in valid_doc_types:
            raise ValueError(f"Invalid doc_type {doc_type!r} in {doc_id}")

        resolved: list[tuple[int, int, str, str, str]] = []
        for entry in note["spans"]:
            sub = str(entry[0])
            etype = str(entry[1])
            assertion = str(entry[2])
            occ = int(entry[3]) if len(entry) >= 4 else 0
            if etype not in valid_types:
                raise ValueError(f"Invalid entity type {etype!r} in {doc_id}")
            if assertion not in valid_assertions:
                raise ValueError(f"Invalid assertion {assertion!r} in {doc_id}")
            if sub != sub.strip() or (sub and sub[-1] in ".,;:?!"):
                raise ValueError(f"Span {sub!r} in {doc_id} has trailing punctuation or whitespace")
            start, end = _find_occurrence(text, sub, occ)
            assert text[start:end] == sub, f"Offset mismatch in {doc_id}: {text[start:end]!r} != {sub!r}"
            resolved.append((start, end, sub, etype, assertion))

        resolved.sort(key=lambda r: (r[0], r[1]))
        for prev, curr in zip(resolved, resolved[1:]):
            if prev[1] > curr[0]:
                raise ValueError(
                    f"Overlapping spans in {doc_id}: {prev[2]!r} [{prev[0]}:{prev[1]}] vs {curr[2]!r} [{curr[0]}:{curr[1]}]"
                )

        results: list[dict[str, Any]] = []
        for r_idx, (start, end, sub, etype, assertion) in enumerate(resolved, start=1):
            region_id = f"{doc_id}_r{r_idx}"
            results.append(
                {
                    "id": region_id,
                    "from_name": "entities",
                    "to_name": "text",
                    "type": "labels",
                    "value": {
                        "start": start,
                        "end": end,
                        "text": sub,
                        "labels": [etype],
                    },
                }
            )
            results.append(
                {
                    "id": region_id,
                    "from_name": "assertion",
                    "to_name": "text",
                    "type": "choices",
                    "value": {
                        "start": start,
                        "end": end,
                        "text": sub,
                        "choices": [assertion],
                    },
                }
            )

        tasks.append(
            {
                "id": task_idx,
                "data": {
                    "doc_id": doc_id,
                    "patient_id": f"pat_{doc_id}",
                    "language": note["language"],
                    "doc_type": doc_type,
                    "document_date": note["document_date"],
                    "authored_by": "llm_single_annotator_not_owner_written",
                    "text": text,
                },
                "annotations": [
                    {
                        "id": task_idx,
                        "result": results,
                    }
                ],
            }
        )
    return tasks


def _masked_5gram_overlap(train_jsonl: Path, eval_texts: list[str]) -> float:
    if not train_jsonl.exists():
        return -1.0

    def norm_5grams(s: str) -> set[tuple[str, ...]]:
        toks = re.sub(r"\d+", "<NUM>", s.lower()).split()
        if len(toks) < 5:
            return set()
        return {tuple(toks[i : i + 5]) for i in range(len(toks) - 4)}

    train_grams: set[tuple[str, ...]] = set()
    for line in train_jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            train_grams |= norm_5grams(json.loads(line)["text"])

    eval_grams: set[tuple[str, ...]] = set()
    for txt in eval_texts:
        eval_grams |= norm_5grams(txt)
    if not eval_grams:
        return 0.0
    return round(len(eval_grams & train_grams) / len(eval_grams), 4)


def main() -> None:
    all_notes = EN_NOTES + FR_NOTES
    tasks = build_label_studio_export(all_notes)

    out_dir = Path("data/processed/realistic_llm")
    out_dir.mkdir(parents=True, exist_ok=True)
    ls_path = out_dir / "label_studio_llm_80.json"
    ls_path.write_text(json.dumps(tasks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Validate through the official strict Label Studio importer with honest provenance tag
    docs = convert_label_studio_records(tasks, source="llm_authored_free_text_single_annotator")
    eval_path = out_dir / "eval.jsonl"
    with eval_path.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")

    lang_counts = Counter(d.language for d in docs)
    dtype_counts = Counter((d.language, d.doc_type) for d in docs)
    etype_counts = Counter(e.type for d in docs for e in d.entities)
    assert_counts = Counter(e.assertion for d in docs for e in d.entities)
    overlap = _masked_5gram_overlap(Path("data/processed/v3_1/train.jsonl"), [d.text for d in docs])

    summary = {
        "total_docs": len(docs),
        "by_language": dict(lang_counts),
        "by_lang_doc_type": {f"{k[0]}:{k[1]}": v for k, v in sorted(dtype_counts.items())},
        "total_entities": sum(etype_counts.values()),
        "entity_types_covered": len(etype_counts),
        "by_entity_type": dict(sorted(etype_counts.items())),
        "assertions_covered": len(assert_counts),
        "by_assertion": dict(sorted(assert_counts.items())),
        "masked_5gram_overlap_with_v3_1_train": overlap,
        "label_studio_export": str(ls_path),
        "eval_jsonl": str(eval_path),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
