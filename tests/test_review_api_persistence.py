from __future__ import annotations

import json
from pathlib import Path

from starlette.testclient import TestClient

from medresearch_platform.config import PlatformConfig
from medresearch_platform.interfaces.api import create_app


def _build_test_config(tmp_path: Path) -> PlatformConfig:
    return PlatformConfig(
        profile="test",
        workspace_dir=tmp_path,
        storage_dir=tmp_path / "storage",
        sqlite_path=tmp_path / "db" / "review_test.sqlite3",
        postgres_url=None,
        max_cpu_threads=2,
        max_memory_mb=1024,
        max_upload_bytes=5 * 1024 * 1024,
        max_document_pages=50,
        default_random_seed=20261112,
        token_secret_key="test_api_secret_key_0123456789",
        pseudonym_vault_key="test_vault_secret_key_0123456789",
    )


def test_review_post_list_reload_and_export_gold(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MEDRESEARCH_DEMO_PASSWORD", "test-demo-pw-2026")
    app = create_app(config=_build_test_config(tmp_path))
    client = TestClient(app)

    login = client.post(
        "/api/v1/auth/token",
        json={"username": "admin_dakar", "password": "test-demo-pw-2026"},
    )
    assert login.status_code == 200
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    seed_resp = client.post("/api/v1/experiments/run", headers=headers)
    assert seed_resp.status_code == 200

    doc_id = "doc_en_subj1_enc1"
    detail = client.get(f"/api/v1/documents/{doc_id}", headers=headers)
    assert detail.status_code == 200
    doc_data = detail.json()
    entities = doc_data["entities"]
    assert len(entities) >= 2

    e0 = entities[0]
    e1 = entities[1]
    raw_text = doc_data["document"].get("raw_text") or doc_data["document"].get("deidentified_preview") or ""
    span_end = min(len(raw_text), max(1, e1["char_start"] + 4))
    fallback_surface = e1.get("source_text") or e1.get("surface_text") or "term"

    # 1. Approve first entity via POST /api/v1/reviews
    r_app = client.post(
        "/api/v1/reviews",
        headers=headers,
        json={"entity_id": e0["entity_id"], "action": "approve", "rationale": "Verified span"},
    )
    assert r_app.status_code == 200
    body_app = r_app.json()
    assert body_app["review_status"] == "APPROVED_GROUND_TRUTH"
    assert body_app["reviewer_user_id"]
    assert body_app["created_at"]

    # 2. Correct second entity (span + type + assertion) via POST /api/v1/reviews
    r_corr = client.post(
        "/api/v1/reviews",
        headers=headers,
        json={
            "entity_id": e1["entity_id"],
            "action": "correct",
            "corrected_entity_type": "SYMPTOM",
            "corrected_assertion_status": "NEGATED",
            "corrected_char_start": e1["char_start"],
            "corrected_char_end": span_end,
            "corrected_surface_text": raw_text[e1["char_start"] : span_end] or fallback_surface,
            "rationale": "Negated in note",
        },
    )
    assert r_corr.status_code == 200
    body_corr = r_corr.json()
    assert body_corr["review_status"] == "HUMAN_CORRECTED"
    assert body_corr["previous_value"]["entity_type"] == e1["entity_type"]
    assert body_corr["corrected_value"]["entity_type"] == "SYMPTOM"
    assert body_corr["corrected_value"]["assertion_status"] == "NEGATED"

    # 3. Reload document detail -> persisted changes survive reload
    reloaded = client.get(f"/api/v1/documents/{doc_id}", headers=headers).json()
    by_id = {e["entity_id"]: e for e in reloaded["entities"]}
    assert by_id[e0["entity_id"]]["review_status"] == "APPROVED_GROUND_TRUTH"
    assert by_id[e1["entity_id"]]["review_status"] == "HUMAN_CORRECTED"
    assert by_id[e1["entity_id"]]["entity_type"] == "SYMPTOM"
    assert by_id[e1["entity_id"]]["assertion_status"] == "NEGATED"

    # 4. List reviews via GET /api/v1/reviews
    r_list = client.get(f"/api/v1/reviews?document_id={doc_id}", headers=headers)
    assert r_list.status_code == 200
    reviews = r_list.json()["reviews"]
    assert len(reviews) == 2

    # 5. Export gold JSONL and BRAT
    r_jsonl = client.get(f"/api/v1/reviews/export?format=jsonl&document_id={doc_id}", headers=headers)
    assert r_jsonl.status_code == 200
    exported_docs = [json.loads(line) for line in r_jsonl.text.splitlines() if line.strip()]
    assert len(exported_docs) == 1
    assert len(exported_docs[0]["entities"]) == 2

    r_brat = client.get(f"/api/v1/reviews/export?format=brat&document_id={doc_id}", headers=headers)
    assert r_brat.status_code == 200
    assert "T1\t" in r_brat.text
    assert "SYMPTOM" in r_brat.text
