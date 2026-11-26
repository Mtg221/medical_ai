"""Phase 8 tests: traceable EN/FR Markdown, HTML, and PDF human-readable clinical reports."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi.testclient import TestClient

from medresearch_platform.config import PlatformConfig
from medresearch_platform.interfaces.api import create_app
from medresearch_platform.reports import generate_report_bytes


def test_bilingual_markdown_html_and_pdf_reports() -> None:
    doc = {
        "document_id": "doc_rep_01",
        "original_filename": "discharge_fr_en.pdf",
        "detected_language": "fr",
        "extraction_mode": "TESSERACT_OCR",
    }
    entities = [
        {
            "entity_id": "e1",
            "surface_text": "douleur thoracique",
            "normalized_text": "chest pain",
            "entity_type": "SYMPTOM",
            "assertion_status": "absent",
            "confidence": 0.94,
            "page_number": 1,
            "char_start": 10,
            "char_end": 28,
        },
        {
            "entity_id": "e2",
            "surface_text": "Metformine",
            "normalized_text": "metformin",
            "entity_type": "MEDICATION",
            "assertion_status": "present",
            "confidence": 0.97,
            "page_number": 1,
            "char_start": 45,
            "char_end": 55,
        },
    ]

    md_fr, mime_md = generate_report_bytes(doc, entities, [], lang="fr", fmt="markdown")
    assert "text/markdown" in mime_md
    md_text = md_fr.decode("utf-8")
    assert "OUTIL DE RECHERCHE UNIQUEMENT" in md_text
    assert "NÉGATION / ABSENT" in md_text
    assert "[doc=doc_rep_01, p.1, chars 10-28]" in md_text
    assert "Aucun élément extrait de la source." in md_text

    html_en, mime_html = generate_report_bytes(doc, entities, [], lang="en", fmt="html")
    assert "text/html" in mime_html
    html_text = html_en.decode("utf-8")
    assert "RESEARCH TOOL ONLY" in html_text
    assert "NEGATED / ABSENT" in html_text

    pdf_bytes, mime_pdf = generate_report_bytes(doc, entities, [], lang="en", fmt="pdf")
    assert mime_pdf == "application/pdf"
    assert pdf_bytes.startswith(b"%PDF-")


def test_api_document_report_endpoint(tmp_path: Path) -> None:
    os.environ["MEDRESEARCH_DEMO_PASSWORD"] = "Phase8-Test-Pass!2026"
    cfg = PlatformConfig(
        profile="test",
        workspace_dir=tmp_path,
        storage_dir=tmp_path / "storage",
        sqlite_path=tmp_path / "db" / "main.sqlite3",
        postgres_url=None,
        max_cpu_threads=2,
        max_memory_mb=1024,
        max_upload_bytes=2 * 1024 * 1024,
        max_document_pages=20,
        default_random_seed=42,
        token_secret_key="phase8-secret",
        pseudonym_vault_key="phase8-vault",
    )
    app = create_app(cfg)
    app.state.service.seed_synthetic_corpus()
    client = TestClient(app)

    tok = client.post(
        "/api/v1/auth/token",
        json={"username": "admin_dakar", "password": os.environ["MEDRESEARCH_DEMO_PASSWORD"]},
    ).json()["access_token"]
    headers = {"Authorization": f"Bearer {tok}"}

    ing = client.post(
        "/api/v1/documents/ingest",
        headers=headers,
        json={
            "filename": "p8_note.txt",
            "text": "CHIEF COMPLAINT: Patient denies chest pain. MEDICATIONS: Metformin 500 mg daily.",
            "subject_id": "subj_p8_01",
        },
    )
    assert ing.status_code == 201
    doc_id = ing.json()["document_id"]

    r_md = client.get(f"/api/v1/documents/{doc_id}/report?format=markdown&lang=fr", headers=headers)
    assert r_md.status_code == 200
    assert "OUTIL DE RECHERCHE UNIQUEMENT" in r_md.text

    r_pdf = client.get(f"/api/v1/documents/{doc_id}/report?format=pdf&lang=en", headers=headers)
    assert r_pdf.status_code == 200
    assert r_pdf.content.startswith(b"%PDF-")
