"""Extract and stage medical_ai_eval_starter.zip for human review, editing, double-annotation, and Label Studio import."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medresearch_platform.data.schema import ASSERTION_CLASSES, DOC_TYPES, ENTITY_TYPES

NOTE_LABEL_MAP: dict[str, str] = {
    "DISEASE": "DIAGNOSIS",
    "PROBLEM": "DIAGNOSIS",
    "TEST": "LAB_TEST",
}

PHI_LABEL_MAP: dict[str, str] = {
    "NAME": "PERSON",
    "ID": "MRN",
}


def _find_non_overlapping_span(
    text: str,
    target: str,
    used: list[tuple[int, int]],
) -> tuple[int, int] | None:
    """Locate exact (or case-insensitive fallback) character offsets for `target` without overlapping `used`."""
    if not target:
        return None
    for hay, needle in ((text, target), (text.lower(), target.lower())):
        start_from = 0
        while True:
            idx = hay.find(needle, start_from)
            if idx == -1:
                break
            cand = (idx, idx + len(target))
            if not any(not (cand[1] <= us or cand[0] >= ue) for us, ue in used):
                return cand
            start_from = idx + 1
    return None


def _load_training_normalized_texts(train_paths: list[Path]) -> set[str]:
    norm_texts: set[str] = set()
    for p in train_paths:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            txt = str(obj.get("text", "")).strip()
            if txt:
                norm_texts.add(re.sub(r"\s+", " ", txt.lower()))
    return norm_texts


def _safe_extract_zip(zip_path: Path, raw_dir: Path) -> dict[str, str]:
    """Safely extract zip archive (blocking path traversal) and return SHA-256 per extracted file."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_resolved = raw_dir.resolve()
    hashes: dict[str, str] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            dest = (raw_resolved / info.filename).resolve()
            if not str(dest).startswith(str(raw_resolved)):
                raise ValueError(f"Blocked unsafe zip path traversal entry: {info.filename}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            data = zf.read(info.filename)
            dest.write_bytes(data)
            hashes[info.filename] = hashlib.sha256(data).hexdigest()
    return hashes


def stage_eval_starter(
    zip_path: Path,
    out_dir: Path = Path("data/staging/eval_starter"),
    *,
    train_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Extract starter zip and produce Label Studio draft tasks (`predictions`, not `annotations`) + PHI staging JSONL."""
    if not zip_path.exists():
        raise FileNotFoundError(f"Starter archive not found: {zip_path}")

    raw_dir = out_dir / "raw"
    file_hashes = _safe_extract_zip(zip_path, raw_dir)

    notes_path = raw_dir / "clinical_notes_80_draft.jsonl"
    phi_path = raw_dir / "phi_sentences_60_draft.jsonl"
    notes_raw = [json.loads(line) for line in notes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    phi_raw = [json.loads(line) for line in phi_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    valid_entities = set(ENTITY_TYPES)
    valid_assertions = set(ASSERTION_CLASSES)

    # 1. Convert 80 clinical note drafts -> Label Studio tasks with editable `predictions` (NOT finalized `annotations`)
    ls_tasks: list[dict[str, Any]] = []
    unmatched_note_spans: list[dict[str, Any]] = []
    lang_counters: Counter[str] = Counter()
    mapped_label_counts: Counter[str] = Counter()

    for idx, row in enumerate(notes_raw, start=1):
        lang = str(row.get("language", "")).strip().lower()
        if lang not in ("en", "fr"):
            raise ValueError(f"Row {idx} has invalid language: {lang!r}")
        lang_counters[lang] += 1
        lang_idx = lang_counters[lang]
        text = str(row["text"])
        doc_type_suggestion = DOC_TYPES[(lang_idx - 1) % len(DOC_TYPES)]

        used_spans: list[tuple[int, int]] = []
        pred_results: list[dict[str, Any]] = []

        for e_idx, ent in enumerate(row.get("entities", []), start=1):
            raw_lbl = str(ent.get("label", "")).strip().upper()
            canon_lbl = NOTE_LABEL_MAP.get(raw_lbl, raw_lbl)
            if canon_lbl not in valid_entities:
                unmatched_note_spans.append({"note_id": row.get("note_id"), "reason": f"unknown_label:{raw_lbl}", "entity": ent})
                continue
            span_txt = str(ent.get("text", ""))
            loc = _find_non_overlapping_span(text, span_txt, used_spans)
            if loc is None:
                unmatched_note_spans.append({"note_id": row.get("note_id"), "reason": "text_not_found", "entity": ent})
                continue
            s, e = loc
            used_spans.append((s, e))
            mapped_label_counts[canon_lbl] += 1
            region_id = f"draft_{idx}_{e_idx}"
            pred_results.append(
                {
                    "id": region_id,
                    "type": "labels",
                    "from_name": "entities",
                    "to_name": "text",
                    "value": {
                        "start": s,
                        "end": e,
                        "text": text[s:e],
                        "labels": [canon_lbl],
                    },
                }
            )
            raw_assert = str(ent.get("assertion") or "").strip().lower()
            if raw_assert and raw_assert in valid_assertions and raw_assert != "present":
                pred_results.append(
                    {
                        "id": region_id,
                        "type": "choices",
                        "from_name": "assertion",
                        "to_name": "text",
                        "value": {"choices": [raw_assert]},
                    }
                )

        ls_tasks.append(
            {
                "id": idx,
                "data": {
                    "doc_id": f"owner_{lang}_{lang_idx:02d}",
                    "starter_note_id": str(row.get("note_id", f"CN{idx:03d}")),
                    "language": lang,
                    "doc_type": doc_type_suggestion,
                    "doc_type_needs_review": True,
                    "document_date": "2025-03-15",
                    "provenance_status": "draft_ai_assisted_requires_human_rewrite_and_adjudication",
                    "text": text,
                },
                "predictions": [
                    {
                        "model_version": "starter_draft_ai_assisted_unverified",
                        "result": pred_results,
                    }
                ],
                "annotations": [],
            }
        )

    ls_out_path = out_dir / "label_studio_tasks_80_draft.json"
    ls_out_path.write_text(json.dumps(ls_tasks, indent=2, ensure_ascii=False), encoding="utf-8")

    # 2. Convert 60 PHI sentence drafts -> eval_redaction_human.py JSONL format for human review
    phi_staged_rows: list[dict[str, Any]] = []
    unmatched_phi_spans: list[dict[str, Any]] = []
    phi_label_counts: Counter[str] = Counter()

    for idx, row in enumerate(phi_raw, start=1):
        lang = str(row.get("language", "")).strip().lower()
        text = str(row["text"])
        used_spans = []
        phi_items: list[dict[str, Any]] = []
        for ent in row.get("entities", []):
            raw_lbl = str(ent.get("label", "")).strip().upper()
            canon_lbl = PHI_LABEL_MAP.get(raw_lbl, raw_lbl)
            span_txt = str(ent.get("text", ""))
            loc = _find_non_overlapping_span(text, span_txt, used_spans)
            if loc is None:
                unmatched_phi_spans.append({"sentence_id": row.get("sentence_id"), "reason": "text_not_found", "entity": ent})
                continue
            s, e = loc
            used_spans.append((s, e))
            phi_label_counts[canon_lbl] += 1
            phi_items.append({"type": canon_lbl, "text": text[s:e], "start": s, "end": e})

        phi_staged_rows.append(
            {
                "id": str(row.get("sentence_id", f"PHI-{idx:02d}")),
                "language": lang,
                "review_status": "draft_ai_assisted_requires_human_review",
                "source": str(row.get("source", "fictional_ai_assisted_draft")),
                "text": text,
                "phi_items": phi_items,
            }
        )

    phi_out_path = out_dir / "phi_60_draft_for_human_review.jsonl"
    phi_out_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in phi_staged_rows) + "\n",
        encoding="utf-8",
    )

    # 3. Leakage check against existing training corpora
    default_trains = [
        Path("data/processed/v3_1/train.jsonl"),
        Path("data/processed/open_clinical/train.jsonl"),
    ]
    train_norm_texts = _load_training_normalized_texts(train_paths if train_paths is not None else default_trains)
    note_train_overlaps = sum(
        1
        for t in ls_tasks
        if re.sub(r"\s+", " ", str(t["data"]["text"]).strip().lower()) in train_norm_texts
    )
    phi_train_overlaps = sum(
        1
        for r in phi_staged_rows
        if re.sub(r"\s+", " ", str(r["text"]).strip().lower()) in train_norm_texts
    )

    report: dict[str, Any] = {
        "staged_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_zip": str(zip_path),
        "source_zip_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
        "extracted_files_sha256": file_hashes,
        "provenance_warning": (
            "STAGING DRAFTS ONLY (AI-assisted fictional starter). Do NOT freeze or evaluate with --owner-written "
            "until independently rewritten/reviewed by humans, double-annotated (>=16 of 80 notes), and adjudicated."
        ),
        "clinical_notes_80": {
            "output_path": str(ls_out_path),
            "n_tasks": len(ls_tasks),
            "languages": dict(sorted(lang_counters.items())),
            "suggested_doc_types": dict(sorted(Counter(t["data"]["doc_type"] for t in ls_tasks).items())),
            "preannotated_entities_total": sum(mapped_label_counts.values()),
            "preannotated_entity_counts": dict(sorted(mapped_label_counts.items())),
            "unmatched_spans": len(unmatched_note_spans),
            "train_corpus_exact_norm_overlaps": note_train_overlaps,
            "stored_in_field": "predictions (annotations=[] so human annotators must explicitly review/submit)",
        },
        "phi_sentences_60": {
            "output_path": str(phi_out_path),
            "n_sentences": len(phi_staged_rows),
            "languages": dict(sorted(Counter(r["language"] for r in phi_staged_rows).items())),
            "phi_items_total": sum(phi_label_counts.values()),
            "phi_label_counts": dict(sorted(phi_label_counts.items())),
            "non_phi_negative_sentences_present": sum(1 for r in phi_staged_rows if not r["phi_items"]),
            "action_required": "Add >= 10 tricky non-PHI clinical sentences (phi_items: []) before freezing.",
            "unmatched_spans": len(unmatched_phi_spans),
            "train_corpus_exact_norm_overlaps": phi_train_overlaps,
        },
        "unmatched_details": {
            "notes": unmatched_note_spans,
            "phi": unmatched_phi_spans,
        },
    }
    report_path = out_dir / "staging_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage medical_ai_eval_starter.zip for Label Studio & human review.")
    parser.add_argument("--zip", type=Path, default=Path("medical_ai_eval_starter.zip"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/staging/eval_starter"))
    args = parser.parse_args()
    rep = stage_eval_starter(args.zip, args.out_dir)
    print(
        f"Staged {rep['clinical_notes_80']['n_tasks']} Label Studio draft tasks -> {rep['clinical_notes_80']['output_path']} "
        f"(pre-annotated entities={rep['clinical_notes_80']['preannotated_entities_total']}, "
        f"unmatched={rep['clinical_notes_80']['unmatched_spans']}, train_overlaps={rep['clinical_notes_80']['train_corpus_exact_norm_overlaps']})"
    )
    print(
        f"Staged {rep['phi_sentences_60']['n_sentences']} PHI draft sentences -> {rep['phi_sentences_60']['output_path']} "
        f"(items={rep['phi_sentences_60']['phi_items_total']}, unmatched={rep['phi_sentences_60']['unmatched_spans']}, "
        f"non_phi_negatives={rep['phi_sentences_60']['non_phi_negative_sentences_present']})"
    )


if __name__ == "__main__":
    main()
