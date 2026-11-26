"""Phase 2 E2E integration tests: 3-axis assertion (certainty/temporality/experiencer) across DB, API, review, and exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from medresearch_platform.config import PlatformConfig
from medresearch_platform.interfaces.api import create_app
from medresearch_platform.reports.generator import (
    render_html_report,
    render_markdown_report,
    render_pdf_report,
)


def test_phase2_three_axis_assertion_db_api_review_and_exports(tmp_path: Path) -> None:
    cfg = PlatformConfig.from_env(workspace_dir=tmp_path)
    app = create_app(cfg)
    app.state.service.init_or_reset_default_users("Phase2Pass#2026")

    now_iso = datetime.now(timezone.utc).isoformat()
    with app.state.service.db.transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO datasets ("
            "dataset_id, institution_id, name, version, source_type, license, consent_basis, "
            "permitted_purpose, deidentification_status, languages_json, modalities_json, "
            "record_count, manifest_sha256, retention_policy, created_at"
            ") VALUES ('ds_p2', 'inst_dakar_primary', 'P2', '1.0', 'SYNTHETIC', 'MIT', "
            "'RESEARCH', 'BENCHMARK', 'SYNTHETIC_NO_PHI', '[\"en\",\"fr\"]', '[\"TXT\"]', 0, 'abc', 'RETAIN', ?)",
            (now_iso,),
        )
        conn.execute(
            "INSERT OR IGNORE INTO subjects (subject_id, institution_id, dataset_id, pseudonym_code, language_primary, consent_status, created_at) "
            "VALUES ('subj_p2', 'inst_dakar_primary', 'ds_p2', 'PAT-P2-001', 'en', 'SYNTHETIC_EXEMPT', ?)",
            (now_iso,),
        )

    client = TestClient(app)
    tok = client.post(
        "/api/v1/auth/token",
        json={"username": "admin_dakar", "password": "Phase2Pass#2026"},
    ).json()["access_token"]
    headers = {"Authorization": f"Bearer {tok}"}

    # 1. Ingest document with orthogonal possible+historical and family-history mentions
    r_ing = client.post(
        "/api/v1/documents/ingest",
        headers=headers,
        json={
            "filename": "p2_assertions.txt",
            "text": "Possible history of asthma. Her mother has diabetes.",
            "dataset_id": "ds_p2",
            "subject_id": "subj_p2",
            "family_id": "fam_p2",
            "encounter_date_iso": "2025-03-15",
        },
    )
    assert r_ing.status_code in (200, 201), r_ing.text
    doc_id = r_ing.json()["document_id"]

    # 2. Verify GET /api/v1/documents/{doc_id} returns certainty/temporality/experiencer from SQLite
    detail = client.get(f"/api/v1/documents/{doc_id}", headers=headers).json()
    ents = detail["entities"]
    assert len(ents) >= 1
    for e in ents:
        assert e["certainty"] in ("confirmed", "possible", "absent")
        assert e["temporality"] in ("current", "historical")
        assert e["experiencer"] in ("patient", "family")

    target_ent = ents[0]
    entity_id = target_ent["entity_id"]

    # 3. Independently correct all 3 axes via POST /api/v1/reviews
    r_rev = client.post(
        "/api/v1/reviews",
        headers=headers,
        json={
            "entity_id": entity_id,
            "action": "correct",
            "corrected_certainty": "absent",
            "corrected_temporality": "historical",
            "corrected_experiencer": "family",
            "rationale": "Verified negated historical family condition",
        },
    )
    assert r_rev.status_code == 200, r_rev.text
    updated = r_rev.json()["updated"]
    assert updated["certainty"] == "absent"
    assert updated["temporality"] == "historical"
    assert updated["experiencer"] == "family"
    assert updated["temporal_status"] == "HISTORICAL"

    # 4. Reload from DB via API and confirm persistence
    detail_after = client.get(f"/api/v1/documents/{doc_id}", headers=headers).json()
    reloaded = next(e for e in detail_after["entities"] if e["entity_id"] == entity_id)
    assert reloaded["certainty"] == "absent"
    assert reloaded["temporality"] == "historical"
    assert reloaded["experiencer"] == "family"

    # 5. Verify JSONL & BRAT exports preserve all 3 axes
    r_jsonl = client.get(f"/api/v1/reviews/export?format=jsonl&document_id={doc_id}", headers=headers)
    assert r_jsonl.status_code == 200
    rows = [json.loads(line) for line in r_jsonl.text.splitlines() if line.strip()]
    assert rows
    exported_ent = next(e for e in rows[0]["entities"] if e["id"] == entity_id)
    assert exported_ent["certainty"] == "absent"
    assert exported_ent["temporality"] == "historical"
    assert exported_ent["experiencer"] == "family"

    r_brat = client.get(f"/api/v1/reviews/export?format=brat&document_id={doc_id}", headers=headers)
    assert r_brat.status_code == 200
    assert "Certainty T1 absent" in r_brat.text
    assert "Temporality T1 historical" in r_brat.text
    assert "Experiencer T1 family" in r_brat.text

    # 6. Verify Markdown, HTML, and PDF report generators include the 3-axis tuple
    doc_meta = detail_after["document"]
    md = render_markdown_report(doc_meta, [reloaded], [], lang="en")
    assert "absent/historical/family" in md
    html_out = render_html_report(doc_meta, [reloaded], [], lang="en")
    assert "absent/historical/family" in html_out
    pdf_bytes = render_pdf_report(doc_meta, [reloaded], [], lang="en")
    assert pdf_bytes.startswith(b"%PDF")
