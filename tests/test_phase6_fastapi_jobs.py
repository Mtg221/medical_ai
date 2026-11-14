"""Phase 6 tests: FastAPI OpenAPI 3.1, Pydantic v2 JSON Schema, multipart upload, SQLite jobs, and SSE."""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi.testclient import TestClient

from medresearch_platform.config import PlatformConfig
from medresearch_platform.extraction.schema_models import (
    CLINICAL_RESEARCH_NOTICE,
    ExtractionResultSchema,
    export_extraction_json_schema,
)
from medresearch_platform.interfaces.api import create_app


def _make_cfg(tmp_path: Path) -> PlatformConfig:
    os.environ["MEDRESEARCH_DEMO_PASSWORD"] = "Phase6-Test-Pass!2026"
    return PlatformConfig(
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
        token_secret_key="phase6-secret-key",
        pseudonym_vault_key="phase6-vault-key",
    )


def test_pydantic_schema_export_and_validation(tmp_path: Path) -> None:
    out = export_extraction_json_schema(tmp_path / "medical_extraction.schema.json")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["title"] == "ExtractionResultSchema"
    assert "records" in data["properties"]

    sample = ExtractionResultSchema.model_validate({
        "source_record_id": "rec_1",
        "document_id": "doc_1",
        "language": "en",
        "model_name": "hybrid_v1",
        "model_version": "1.0.0",
        "records": [],
    })
    assert sample.clinical_notice == CLINICAL_RESEARCH_NOTICE


def test_fastapi_openapi_upload_jobs_and_sse(tmp_path: Path) -> None:
    app = create_app(_make_cfg(tmp_path))
    app.state.service.seed_synthetic_corpus()
    client = TestClient(app)

    # OpenAPI 3.1 and schema endpoint
    openapi = client.get("/api/v1/openapi.json").json()
    assert openapi["openapi"].startswith("3.1")
    assert "/api/v1/documents/upload" in openapi["paths"]
    assert "/api/v1/jobs/{job_id}/events" in openapi["paths"]

    schema_resp = client.get("/api/v1/schema/extraction")
    assert schema_resp.status_code == 200
    assert schema_resp.json()["title"] == "ExtractionResultSchema"

    # Authenticate
    tok_resp = client.post(
        "/api/v1/auth/token",
        json={"username": "admin_dakar", "password": os.environ["MEDRESEARCH_DEMO_PASSWORD"]},
    )
    assert tok_resp.status_code == 200
    token = tok_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Multipart upload -> 202 Accepted + background execution
    up_resp = client.post(
        "/api/v1/documents/upload",
        headers=headers,
        files={"file": ("note_phase6.txt", b"CHIEF COMPLAINT: Chest pain.\nMEDICATIONS: Aspirin 81 mg daily.", "text/plain")},
        data={"subject_id": "subj_p6_01", "specialty": "CARDIOLOGY"},
    )
    assert up_resp.status_code == 202
    job_id = up_resp.json()["job_id"]

    # Job status in SQLite
    job_resp = client.get(f"/api/v1/jobs/{job_id}", headers=headers)
    assert job_resp.status_code == 200
    job_data = job_resp.json()
    assert job_data["status"] == "SUCCEEDED"
    assert job_data["progress_pct"] == 100
    assert job_data["current_stage"] == "COMPLETED"
    assert job_data["result"]["document_id"].startswith("doc_")

    # SSE stream
    sse_resp = client.get(f"/api/v1/jobs/{job_id}/events", headers=headers)
    assert sse_resp.status_code == 200
    assert "text/event-stream" in sse_resp.headers["content-type"]
    assert "event: job_update" in sse_resp.text
    assert '"status": "SUCCEEDED"' in sse_resp.text
