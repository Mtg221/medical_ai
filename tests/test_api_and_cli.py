"""End-to-end integration tests for the Starlette REST API, Accessible Web UI, and CLI."""

from __future__ import annotations

from pathlib import Path

from starlette.testclient import TestClient

from medresearch_platform.config import PlatformConfig
from medresearch_platform.interfaces.api import create_app
from medresearch_platform.interfaces.cli import main as cli_main


def _build_test_config(tmp_path: Path) -> PlatformConfig:
    return PlatformConfig(
        profile="test",
        workspace_dir=tmp_path,
        storage_dir=tmp_path / "storage",
        sqlite_path=tmp_path / "db" / "api_test.sqlite3",
        postgres_url=None,
        max_cpu_threads=2,
        max_memory_mb=1024,
        max_upload_bytes=5 * 1024 * 1024,
        max_document_pages=50,
        default_random_seed=20261112,
        token_secret_key="test_api_secret_key_0123456789",
        pseudonym_vault_key="test_vault_secret_key_0123456789",
    )


def test_full_api_lifecycle_rbac_and_institution_isolation(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MEDRESEARCH_DEMO_PASSWORD", "test-demo-pw-2026")
    cfg = _build_test_config(tmp_path)
    app = create_app(config=cfg)
    client = TestClient(app)

    # 1. Health, OpenAPI, and Web UI
    r_health = client.get("/api/v1/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    r_openapi = client.get("/api/v1/openapi.json")
    assert r_openapi.status_code == 200
    assert r_openapi.json()["openapi"] == "3.1.0"

    r_web = client.get("/")
    assert r_web.status_code == 200
    assert "Skip to main content" in r_web.text
    assert "NOT FOR AUTONOMOUS CLINICAL DIAGNOSIS" in r_web.text

    # 2. Unauthenticated access to medical documents must fail (403)
    assert client.get("/api/v1/documents").status_code == 403

    # 3. Issue tokens for Admin (Dakar), Annotator (Dakar), and Partner Researcher (Paris)
    tok_admin = client.post(
        "/api/v1/auth/token",
        json={"username": "admin_dakar", "password": "test-demo-pw-2026"},
    ).json()["access_token"]

    tok_annotator = client.post(
        "/api/v1/auth/token",
        json={"username": "annotator_clinical", "password": "test-demo-pw-2026"},
    ).json()["access_token"]

    tok_paris = client.post(
        "/api/v1/auth/token",
        json={"username": "researcher_paris", "password": "test-demo-pw-2026"},
    ).json()["access_token"]

    # 4. Annotator cannot trigger model training/experiments (RBAC separation)
    r_denied_exp = client.post(
        "/api/v1/experiments/run",
        headers={"Authorization": f"Bearer {tok_annotator}"},
    )
    assert r_denied_exp.status_code == 403

    # 5. Admin triggers Phase 1 experiment & corpus seeding
    r_exp = client.post(
        "/api/v1/experiments/run",
        headers={"Authorization": f"Bearer {tok_admin}"},
    )
    assert r_exp.status_code == 200
    exp_data = r_exp.json()
    assert exp_data["split_leakage_verification"]["leak_detected"] is False
    assert exp_data["metrics"]["ocr_clean"]["cer"] == 0.0
    assert exp_data["metrics"]["nlp_spans"]["exact_f1"] >= 0.65

    # 6. List documents and inspect detail
    r_docs = client.get(
        "/api/v1/documents",
        headers={"Authorization": f"Bearer {tok_admin}"},
    )
    assert r_docs.status_code == 200
    assert r_docs.json()["count"] == 4

    r_detail = client.get(
        "/api/v1/documents/doc_en_subj1_enc1",
        headers={"Authorization": f"Bearer {tok_admin}"},
    )
    assert r_detail.status_code == 200
    detail_json = r_detail.json()
    assert len(detail_json["entities"]) >= 8
    assert len(detail_json["relations"]) >= 2

    # 7. Cross-institution isolation: Paris researcher cannot read Dakar document
    r_cross = client.get(
        "/api/v1/documents/doc_en_subj1_enc1",
        headers={"Authorization": f"Bearer {tok_paris}"},
    )
    assert r_cross.status_code == 403

    # 8. Annotator submits human review on an AI-extracted entity
    first_entity_id = detail_json["entities"][0]["entity_id"]
    r_rev = client.post(
        f"/api/v1/entities/{first_entity_id}/review",
        headers={"Authorization": f"Bearer {tok_annotator}"},
        json={
            "review_decision": "APPROVE_AS_GROUND_TRUTH",
            "rationale": "Verified against admission note section.",
        },
    )
    assert r_rev.status_code == 200
    assert r_rev.json()["updated"]["review_status"] == "APPROVED_GROUND_TRUTH"

    # 9. Longitudinal timeline & contradiction detection across encounters for subj_synth_001
    r_tl = client.get(
        "/api/v1/subjects/subj_synth_001/timeline",
        headers={"Authorization": f"Bearer {tok_admin}"},
    )
    assert r_tl.status_code == 200
    tl_json = r_tl.json()
    assert len(tl_json["timeline_events"]) >= 8
    assert tl_json["contradictions_count"] >= 1

    # 10. Search, Terminology, Models, and Audit Chain Verification
    r_search = client.get(
        "/api/v1/search?q=metformin+hypertension",
        headers={"Authorization": f"Bearer {tok_admin}"},
    )
    assert r_search.status_code == 200
    assert r_search.json()["count"] >= 1

    r_term = client.get("/api/v1/terminology/map?term=diabète+de+type+2&category=DIAGNOSIS")
    assert r_term.status_code == 200
    assert r_term.json()["code"] == "E11.9"

    r_models = client.get("/api/v1/models")
    assert r_models.status_code == 200
    assert r_models.json()["count"] == 3

    r_audit = client.get(
        "/api/v1/audit/verify",
        headers={"Authorization": f"Bearer {tok_admin}"},
    )
    assert r_audit.status_code == 200
    assert r_audit.json()["valid"] is True


def test_frontend_assets_and_security_headers(tmp_path: Path) -> None:
    client = TestClient(create_app(config=_build_test_config(tmp_path)))

    page = client.get("/")
    assert page.status_code == 200
    assert "script-src 'self'" in page.headers["content-security-policy"]
    assert page.headers["x-content-type-options"] == "nosniff"
    assert '<script src="/static/app.js"' in page.text

    js = client.get("/static/app.js")
    assert js.status_code == 200
    assert "innerHTML" not in js.text
    assert client.get("/static/app.css").status_code == 200
    assert client.get("/static/does_not_exist.js").status_code == 404

    react_dist_index = Path(__file__).resolve().parents[1] / "frontend" / "dist" / "index.html"
    react_page = client.get("/react")
    if react_dist_index.exists():
        assert react_page.status_code == 200
        assert "script-src 'self'" in react_page.headers["content-security-policy"]
        assert "/react/assets/" in react_page.text
    else:
        assert react_page.status_code == 404
        assert "React frontend not built" in react_page.json()["error"]


def test_cli_commands_execution(capsys) -> None:
    rc_diag = cli_main(["diagnostics"])
    assert rc_diag == 0
    out_diag = capsys.readouterr().out
    assert "CPU-ONLY (Native)" in out_diag

    rc_audit = cli_main(["verify-audit"])
    assert rc_audit == 0
    out_audit = capsys.readouterr().out
    assert '"valid": true' in out_audit
