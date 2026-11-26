"""Span-preserving OCR-noise data augmentation for robust clinical NER training."""

from __future__ import annotations

import numpy as np

from medresearch_platform.data.schema import (
    AnnotatedDocument,
    EntitySpan,
    TimexSpan,
)

# Strictly 1-to-1 character substitutions so all entity and timex character offsets stay 100% exact
_OCR_CHAR_MAP: dict[str, tuple[str, ...]] = {
    "l": ("1", "I"),
    "I": ("l", "1"),
    "O": ("0",),
    "0": ("O",),
    "é": ("e",),
    "è": ("e",),
    "ê": ("e",),
    "à": ("a",),
    "â": ("a",),
    "ô": ("o",),
    "î": ("i",),
    "ù": ("u",),
    "û": ("u",),
    "ç": ("c",),
    "É": ("E",),
    "È": ("E",),
    "À": ("A",),
    "S": ("5",),
    "B": ("8",),
    "g": ("q",),
    "c": ("e",),
    "e": ("c",),
}


def augment_document_ocr_noise(
    doc: AnnotatedDocument,
    rng: np.random.Generator,
    *,
    char_error_rate: float = 0.04,
    accent_drop_prob: float = 0.35,
    suffix: str = "ocr_aug",
) -> AnnotatedDocument:
    """Create a 1-to-1 length-preserving OCR-perturbed copy of `doc` with exact entity/timex spans."""
    chars = list(doc.text)
    for i, ch in enumerate(chars):
        if ch in {"é", "è", "ê", "à", "â", "ô", "î", "ù", "û", "ç", "É", "È", "À"}:
            if float(rng.random()) < accent_drop_prob:
                chars[i] = _OCR_CHAR_MAP[ch][0]
            continue
        if ch in _OCR_CHAR_MAP and float(rng.random()) < char_error_rate:
            choices = _OCR_CHAR_MAP[ch]
            chars[i] = choices[int(rng.integers(0, len(choices)))]

    new_text = "".join(chars)
    new_entities = [
        EntitySpan(
            id=e.id,
            type=e.type,
            start=e.start,
            end=e.end,
            text=new_text[e.start : e.end],
            assertion=e.assertion,
            temporality=e.temporality,
            normalized_code=e.normalized_code,
            normalized_system=e.normalized_system,
        )
        for e in doc.entities
    ]
    new_timex = [
        TimexSpan(
            id=t.id,
            type=t.type,
            start=t.start,
            end=t.end,
            text=new_text[t.start : t.end],
            normalized=t.normalized,
        )
        for t in doc.timex
    ]
    return AnnotatedDocument(
        doc_id=f"{doc.doc_id}__{suffix}",
        patient_id=doc.patient_id,
        language=doc.language,
        doc_type=doc.doc_type,
        document_date=doc.document_date,
        source=f"{doc.source}_ocr_aug",
        text=new_text,
        entities=new_entities,
        relations=list(doc.relations),
        timex=new_timex,
        split=doc.split,
        metadata={**doc.metadata, "ocr_augmented": True},
    )


def build_ocr_augmented_train_set(
    train_docs: list[AnnotatedDocument],
    *,
    ratio: float = 0.25,
    char_error_rate: float = 0.04,
    seed: int = 42,
) -> list[AnnotatedDocument]:
    """Return `train_docs` plus a deterministic subset of span-preserving OCR-perturbed copies."""
    if ratio <= 0.0 or not train_docs:
        return list(train_docs)
    rng = np.random.default_rng(seed)
    n_aug = max(1, int(round(len(train_docs) * min(ratio, 1.0))))
    idxs = rng.choice(len(train_docs), size=min(n_aug, len(train_docs)), replace=False)
    augmented = list(train_docs)
    for idx in sorted(int(i) for i in idxs):
        augmented.append(
            augment_document_ocr_noise(
                train_docs[idx],
                rng,
                char_error_rate=char_error_rate,
                suffix=f"ocr{idx}",
            )
        )
    return augmented
