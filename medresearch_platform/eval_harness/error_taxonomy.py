"""Automatic error taxonomy classifier for extraction evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from medresearch_platform.data.schema import AnnotatedDocument
from medresearch_platform.eval_harness.metrics import DocumentPrediction

ERROR_CATEGORIES: tuple[str, ...] = (
    "boundary_error",
    "type_confusion",
    "missed_fn",
    "spurious_fp",
    "assertion_wrong",
    "relation_wrong",
    "ocr_induced",
)


@dataclass(frozen=True)
class ErrorExample:
    doc_id: str
    language: str
    doc_type: str
    category: str
    gold: str | None
    predicted: str | None
    context: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) < min(a1, b1)


def _snippet(text: str, start: int, end: int, pad: int = 40) -> str:
    lo = max(0, start - pad)
    hi = min(len(text), end + pad)
    return text[lo:hi].replace("\n", " ")


def classify_document_errors(gold: AnnotatedDocument, pred: DocumentPrediction) -> list[ErrorExample]:
    """Assign error taxonomy categories for one document."""
    errors: list[ErrorExample] = []
    matched_gold: set[int] = set()
    matched_pred: set[int] = set()

    # 1. Exact span + type matches -> check assertion
    for gi, ge in enumerate(gold.entities):
        for pi, pe in enumerate(pred.entities):
            if pi in matched_pred:
                continue
            if ge.start == pe.start and ge.end == pe.end and ge.type == pe.type:
                matched_gold.add(gi)
                matched_pred.add(pi)
                if ge.assertion != pe.assertion:
                    errors.append(
                        ErrorExample(
                            doc_id=gold.doc_id,
                            language=gold.language,
                            doc_type=gold.doc_type,
                            category="assertion_wrong",
                            gold=f"{ge.type}:{ge.text} [{ge.assertion}]",
                            predicted=f"{pe.type}:{pe.text} [{pe.assertion}]",
                            context=_snippet(gold.text, ge.start, ge.end),
                        )
                    )
                break

    # 2. Overlapping spans -> boundary_error (same type) or type_confusion (different type)
    for gi, ge in enumerate(gold.entities):
        if gi in matched_gold:
            continue
        for pi, pe in enumerate(pred.entities):
            if pi in matched_pred:
                continue
            if _overlaps(ge.start, ge.end, pe.start, pe.end):
                matched_gold.add(gi)
                matched_pred.add(pi)
                cat = "boundary_error" if ge.type == pe.type else "type_confusion"
                errors.append(
                    ErrorExample(
                        doc_id=gold.doc_id,
                        language=gold.language,
                        doc_type=gold.doc_type,
                        category=cat,
                        gold=f"{ge.type}:{ge.text} ({ge.start}:{ge.end})",
                        predicted=f"{pe.type}:{pe.text} ({pe.start}:{pe.end})",
                        context=_snippet(gold.text, min(ge.start, pe.start), max(ge.end, pe.end)),
                    )
                )
                break

    # 3. Remaining unmatched gold -> missed_fn (or ocr_induced if OCR was used)
    for gi, ge in enumerate(gold.entities):
        if gi in matched_gold:
            continue
        errors.append(
            ErrorExample(
                doc_id=gold.doc_id,
                language=gold.language,
                doc_type=gold.doc_type,
                category="ocr_induced" if pred.ocr_induced else "missed_fn",
                gold=f"{ge.type}:{ge.text}",
                predicted=None,
                context=_snippet(gold.text, ge.start, ge.end),
            )
        )

    # 4. Remaining unmatched pred -> spurious_fp
    for pi, pe in enumerate(pred.entities):
        if pi in matched_pred:
            continue
        errors.append(
            ErrorExample(
                doc_id=gold.doc_id,
                language=gold.language,
                doc_type=gold.doc_type,
                category="spurious_fp",
                gold=None,
                predicted=f"{pe.type}:{pe.text}",
                context=_snippet(gold.text, pe.start, pe.end),
            )
        )

    # 5. Relations
    ent_by_id = {e.id: e for e in gold.entities}
    gold_rels = {
        (r.type, ent_by_id[r.head].start, ent_by_id[r.head].end, ent_by_id[r.tail].start, ent_by_id[r.tail].end)
        for r in gold.relations
        if r.head in ent_by_id and r.tail in ent_by_id
    }
    pred_rels = {(r.type, r.head_start, r.head_end, r.tail_start, r.tail_end) for r in pred.relations}
    for rel in sorted(gold_rels - pred_rels):
        errors.append(
            ErrorExample(
                doc_id=gold.doc_id,
                language=gold.language,
                doc_type=gold.doc_type,
                category="relation_wrong",
                gold=f"{rel[0]} ({rel[1]}:{rel[2]} -> {rel[3]}:{rel[4]})",
                predicted=None,
                context=_snippet(gold.text, min(rel[1], rel[3]), max(rel[2], rel[4])),
            )
        )

    return errors


def classify_corpus_errors(
    gold_docs: list[AnnotatedDocument],
    predictions: list[DocumentPrediction],
) -> list[ErrorExample]:
    pred_by_id = {p.doc_id: p for p in predictions}
    all_errors: list[ErrorExample] = []
    for g in gold_docs:
        p = pred_by_id.get(g.doc_id, DocumentPrediction(doc_id=g.doc_id, language=g.language, doc_type=g.doc_type))
        all_errors.extend(classify_document_errors(g, p))
    return all_errors
