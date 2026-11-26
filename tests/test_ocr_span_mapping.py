"""Regression tests for character-level OCR-to-reference span mapping (H3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from medresearch_platform.documents.ocr_benchmark import map_ocr_spans_to_reference
from medresearch_platform.eval_harness.runner import _load_jsonl


@pytest.mark.parametrize("split", ["val", "test_iid", "test_unseen_terms", "test_unseen_templates"])
def test_identical_text_mapping_is_100_percent_exact(split: str) -> None:
    docs = _load_jsonl(Path(f"data/processed/v3_1/{split}.jsonl"))[:100]
    assert len(docs) == 100
    for doc in docs:
        gold_spans = [(e.start, e.end) for e in doc.entities]
        mapped = map_ocr_spans_to_reference(doc.text, doc.text, gold_spans)
        assert mapped == gold_spans, f"Exact identity failed on {doc.doc_id} ({split})"


def test_small_ocr_noise_mapping_preserves_exact_reference_bounds() -> None:
    # 1. Punctuation touching entity + extra space
    ref = "Started metformin 500 mg, fever absent."
    ocr_extra_space = "Started  metformin 500 mg, fever absent."
    # Span for "500 mg" in ocr_extra_space
    s1 = ocr_extra_space.index("500 mg")
    e1 = s1 + len("500 mg")
    # Span for "fever" in ocr_extra_space
    s2 = ocr_extra_space.index("fever")
    e2 = s2 + len("fever")
    mapped = map_ocr_spans_to_reference(ocr_extra_space, ref, [(s1, e1), (s2, e2)])
    assert mapped == [
        (ref.index("500 mg"), ref.index("500 mg") + len("500 mg")),
        (ref.index("fever"), ref.index("fever") + len("fever")),
    ]

    # 2. One replaced character inside an entity ("metfornin" vs "metformin")
    ocr_char_err = "Started metfornin 500 mg, fever absent."
    s_err = ocr_char_err.index("metfornin")
    e_err = s_err + len("metfornin")
    mapped_err = map_ocr_spans_to_reference(ocr_char_err, ref, [(s_err, e_err)])
    assert mapped_err == [(ref.index("metformin"), ref.index("metformin") + len("metformin"))]

    # 3. Merged line (newline replaced by space)
    ref_lines = "Patient has pneumonia.\nStarted amoxicillin 1 g."
    ocr_merged = "Patient has pneumonia. Started amoxicillin 1 g."
    s_p = ocr_merged.index("pneumonia")
    e_p = s_p + len("pneumonia")
    s_a = ocr_merged.index("amoxicillin")
    e_a = s_a + len("amoxicillin")
    mapped_lines = map_ocr_spans_to_reference(ocr_merged, ref_lines, [(s_p, e_p), (s_a, e_a)])
    assert mapped_lines == [
        (ref_lines.index("pneumonia"), ref_lines.index("pneumonia") + len("pneumonia")),
        (ref_lines.index("amoxicillin"), ref_lines.index("amoxicillin") + len("amoxicillin")),
    ]
