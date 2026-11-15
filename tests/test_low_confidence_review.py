"""Tests for K4: low-confidence fact gating, persistence, and report separation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from medresearch_platform.eval_harness.metrics import DocumentPrediction
from medresearch_platform.reports.generator import render_html_report, render_markdown_report
from medresearch_platform.serving.extractor import ExtractionService


def test_low_confidence_entity_flagged_needs_review(tmp_path: Path) -> None:
    svc = ExtractionService(model_dir=tmp_path / "dummy_hybrid")
    svc._hybrid = SimpleNamespace(  # type: ignore[assignment]
        manifest={
            "model_type": "hybrid",
            "seed": 1,
            "cfg": {"review_threshold": 0.60},
            "corpus_id": "synth_v3_1",
            "corpus_hash": "abc123",
            "git_commit": "test",
            "files": {},
        },
        seed=1,
        cfg={"review_threshold": 0.60},
        extract_document=lambda doc: (
            DocumentPrediction(
                doc_id=doc.doc_id,
                language=doc.language,
                doc_type="discharge_summary",
                predicted_doc_type="discharge_summary",
                entities=[
                    SimpleNamespace(
                        type="DIAGNOSIS",
                        start=0,
                        end=12,
                        text="hypertension",
                        assertion="present",
                        confidence=0.05,
                    ),
                    SimpleNamespace(
                        type="MEDICATION",
                        start=20,
                        end=30,
                        text="metformine",
                        assertion="present",
                        confidence=0.92,
                    ),
                ],
                relations=[],
                timex=[],
            ),
            {},
        ),
    )

    res = svc.extract("hypertension ... metformine", document_id="doc_k4", language="fr")
    assert len(res.records) == 2

    low_rec = res.records[0]
    high_rec = res.records[1]

    assert low_rec.confidence_score == 0.05
    assert low_rec.provenance["review_status"] == "needs_review"
    assert low_rec.provenance["review_reason"] == "low_confidence"
    assert low_rec.provenance["review_threshold"] == 0.60

    assert high_rec.confidence_score == 0.92
    assert high_rec.provenance["review_status"] == "AI_PREDICTED"
    assert high_rec.provenance["review_reason"] == ""


def test_reports_exclude_needs_review_from_confirmed_counts_and_render_to_verify() -> None:
    doc = {
        "document_id": "doc_rep_k4",
        "original_filename": "note.txt",
        "detected_language": "en",
        "extraction_mode": "NATIVE_TEXT",
    }
    entities = [
        {
            "entity_id": "e1",
            "entity_type": "MEDICATION",
            "source_text": "metformin",
            "normalized_text": "metformin",
            "assertion_status": "present",
            "confidence": 0.95,
            "page_number": 1,
            "char_start": 0,
            "char_end": 9,
            "review_status": "AI_PREDICTED",
            "review_reason": "",
        },
        {
            "entity_id": "e2",
            "entity_type": "LAB_TEST",
            "source_text": "Toux productive",
            "normalized_text": "Toux productive",
            "assertion_status": "present",
            "confidence": 0.05,
            "page_number": 1,
            "char_start": 15,
            "char_end": 30,
            "review_status": "needs_review",
            "review_reason": "low_confidence",
        },
    ]

    md_en = render_markdown_report(doc, entities, [], lang="en")
    assert "**Total Extracted Facts:** 1" in md_en
    assert "## To Verify (Low Confidence)" in md_en
    assert "Toux productive" in md_en.split("## To Verify (Low Confidence)")[1]
    assert "Toux productive" not in md_en.split("## To Verify (Low Confidence)")[0]

    html_en = render_html_report(doc, entities, [], lang="en")
    assert "<strong>Total Extracted Facts:</strong> 1" in html_en
    assert "To Verify (Low Confidence)" in html_en

    md_fr = render_markdown_report(doc, entities, [], lang="fr")
    assert "**Nombre total de faits extraits:** 1" in md_fr
    assert "## À vérifier (faible confiance)" in md_fr
