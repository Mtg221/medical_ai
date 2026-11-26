from __future__ import annotations

from medresearch_platform.reports.generator import render_html_report, render_markdown_report


def test_report_hides_heatmap_and_suppresses_clinical_risk_band() -> None:
    doc = {"document_id": "doc_test", "original_filename": "note.txt", "detected_language": "en"}
    findings = [
        {
            "image_id": "img_clin_01",
            "picture_type": "skin_clinical",
            "malignancy_risk_band": "low",
            "malignancy_risk_probability": 0.18,
            "review_status": "needs_review",
            "body_site": "left forearm",
            "abstained": False,
            "top3_diagnoses": [{"diagnosis": "nevus", "probability": 0.72}],
        },
        {
            "image_id": "img_derm_02",
            "picture_type": "skin_dermoscopic",
            "malignancy_risk_band": "high",
            "malignancy_risk_probability": 0.82,
            "review_status": "needs_review",
            "body_site": "anterior torso",
            "abstained": False,
            "top3_diagnoses": [{"diagnosis": "melanoma", "probability": 0.65}],
        },
    ]

    md_en = render_markdown_report(doc, [], [], image_findings=findings, lang="en")
    assert "Grad-CAM heat map omitted from report" in md_en
    assert "No risk band shown (phone photo: served sensitivity 21.6% < 50% — please see a dermatologist)" in md_en
    assert "Risk band: **`HIGH`**" in md_en

    md_fr = render_markdown_report(doc, [], [], image_findings=findings, lang="fr")
    assert "Carte de chaleur Grad-CAM omise du rapport" in md_fr
    assert (
        "Aucune catégorie de risque affichée (photo par téléphone : sensibilité servie 21,6 % < 50 % — veuillez consulter un dermatologue)"
        in md_fr
    )

    html_en = render_html_report(doc, [], [], image_findings=findings, lang="en")
    assert "Grad-CAM heat map omitted from report" in html_en
    assert "No risk band shown" in html_en
