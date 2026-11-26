"""Phase 1 failing-first regression tests for P1 through P9."""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from medresearch_platform.config import PlatformConfig
from medresearch_platform.data.cli import main as data_cli_main
from medresearch_platform.data.schema import AnnotatedDocument
from medresearch_platform.hybrid.pipeline import HybridPipeline
from medresearch_platform.interfaces.api import create_app
from medresearch_platform.serving.extractor import ExtractionService
from medresearch_platform.workflows.pipeline import MedicalResearchPlatformService


def _get_auth_headers(client: TestClient, username: str = "admin_dakar", password: str = "ChangeMe_LocalDev_2026!") -> dict[str, str]:
    resp = client.post("/api/v1/auth/token", json={"username": username, "password": password})
    assert resp.status_code == 200, f"Login failed for {username}: {resp.text}"
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_p1_cli_uses_saved_model_and_matches_api_and_hybrid(
    tmp_path: Path,
    isolated_test_models: tuple[Path, Path],
) -> None:
    """P1: CLI extract & report must load the saved model via ExtractionService and match API + HybridPipeline."""
    hybrid_dir, _ = isolated_test_models
    note_text = (
        "Date : 14/11/2026\n"
        "Compte rendu d'hospitalisation.\n"
        "Diagnostic : pneumonie communautaire. Traitement : amoxicilline 1000 mg deux fois par jour.\n"
        "Pas de fièvre ni de douleur thoracique.\n"
    )
    note_file = tmp_path / "sample_fr.txt"
    note_file.write_text(note_text, encoding="utf-8", newline="\n")
    out_json = tmp_path / "cli_out.json"

    # Missing model should fail with clear train instruction
    with pytest.raises((FileNotFoundError, SystemExit)):
        data_cli_main(["extract", str(note_file), "--model", str(tmp_path / "missing_model"), "--out", str(out_json)])

    # Valid model should load saved model without training a throwaway one
    rc = data_cli_main(["extract", str(note_file), "--model", str(hybrid_dir), "--out", str(out_json)])
    assert rc == 0
    cli_payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert cli_payload.get("model_name") == "hybrid_v2"
    assert str(cli_payload.get("model_version", "")).startswith("seed")

    svc = ExtractionService(model_dir=hybrid_dir)
    svc_res = svc.extract(note_text, language="fr", document_date="2026-11-14")
    cli_spans = [
        (r["entity_type"], r["char_start"], r["char_end"], r["assertion_status"])
        for r in cli_payload["records"]
    ]
    svc_spans = [
        (r.entity_type, r.char_start, r.char_end, r.assertion_status)
        for r in svc_res.records
    ]
    assert cli_spans == svc_spans

    direct_pipe = HybridPipeline.load(hybrid_dir)
    pred, _ = direct_pipe.extract_document(
        AnnotatedDocument(
            doc_id="direct",
            patient_id="PAT-1",
            language="fr",
            doc_type="discharge_summary",
            document_date="2026-11-14",
            source="direct",
            text=note_text,
        )
    )
    assert [(e.type, e.start, e.end) for e in pred.entities] == [
        (r["entity_type"], r["char_start"], r["char_end"]) for r in cli_payload["records"]
    ]


def test_p2_users_cli_and_login_after_existing_db(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_test_models: tuple[Path, Path],
) -> None:
    """P2: Even if DB was created first without MEDRESEARCH_DEMO_PASSWORD, setting it or running users init allows login."""
    hybrid_dir, crf_dir = isolated_test_models
    db_file = tmp_path / "p2.sqlite3"
    monkeypatch.setenv("MEDRESEARCH_DB_PATH", str(db_file))
    monkeypatch.setenv("MEDRESEARCH_STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("MEDRESEARCH_MODEL_DIR", str(hybrid_dir))
    monkeypatch.setenv("MEDRESEARCH_CLASSICAL_CRF_DIR", str(crf_dir))
    monkeypatch.delenv("MEDRESEARCH_DEMO_PASSWORD", raising=False)

    # First run without MEDRESEARCH_DEMO_PASSWORD (simulating running tests or CLI first)
    cfg1 = PlatformConfig.from_env()
    assert cfg1.sqlite_path == db_file
    svc1 = MedicalResearchPlatformService(cfg1)
    svc1.initialize_platform()

    # Now user starts server with MEDRESEARCH_DEMO_PASSWORD set
    demo_pw = "ChangeMe_LocalDev_2026!"
    monkeypatch.setenv("MEDRESEARCH_DEMO_PASSWORD", demo_pw)
    app = create_app(PlatformConfig.from_env())
    client = TestClient(app)

    for username in ("admin_dakar", "researcher_dakar", "annotator_dakar", "ml_dakar", "partner_paris"):
        resp = client.post("/api/v1/auth/token", json={"username": username, "password": demo_pw})
        assert resp.status_code == 200, f"Expected 200 login for {username}, got {resp.status_code}: {resp.text}"

    # Also test `medextract users set-password` CLI subcommand
    rc = data_cli_main(["users", "set-password", "admin_dakar", "--password", "NewCustomPass_2026!"])
    assert rc == 0
    resp2 = client.post("/api/v1/auth/token", json={"username": "admin_dakar", "password": "NewCustomPass_2026!"})
    assert resp2.status_code == 200


def test_p3_scan_without_tesseract_fails_loudly_and_empty_scan_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_test_models: tuple[Path, Path],
) -> None:
    """P3: Scans without Tesseract must fail with a clear error message, never SUCCEEDED with 0 entities."""
    hybrid_dir, crf_dir = isolated_test_models
    monkeypatch.setenv("MEDRESEARCH_DB_PATH", str(tmp_path / "p3.sqlite3"))
    monkeypatch.setenv("MEDRESEARCH_SQLITE_PATH", str(tmp_path / "p3.sqlite3"))
    monkeypatch.setenv("MEDRESEARCH_STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("MEDRESEARCH_MODEL_DIR", str(hybrid_dir))
    monkeypatch.setenv("MEDRESEARCH_CLASSICAL_CRF_DIR", str(crf_dir))
    monkeypatch.setenv("MEDRESEARCH_DEMO_PASSWORD", "ChangeMe_LocalDev_2026!")

    import medresearch_platform.documents.ocr_engine as ocr_mod
    import medresearch_platform.documents.parsers as parsers_mod

    monkeypatch.setattr(ocr_mod, "check_tesseract_available", lambda: (False, []))
    if hasattr(parsers_mod, "check_tesseract_available"):
        monkeypatch.setattr(parsers_mod, "check_tesseract_available", lambda: (False, []))

    app = create_app(PlatformConfig.from_env())
    client = TestClient(app)
    headers = _get_auth_headers(client)

    health = client.get("/api/v1/health")
    assert health.status_code == 200
    assert "ocr" in health.json()
    assert health.json()["ocr"]["tesseract_available"] is False

    img = Image.new("RGB", (400, 200), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    up = client.post(
        "/api/v1/documents/upload",
        headers=headers,
        files={"file": ("scan_note.png", png_bytes, "image/png")},
    )
    assert up.status_code == 202
    job_id = up.json()["job_id"]
    job_resp = client.get(f"/api/v1/jobs/{job_id}", headers=headers)
    assert job_resp.status_code == 200
    job_data = job_resp.json()
    assert job_data["status"] == "FAILED"
    assert "tesseract" in str(job_data.get("error") or "").lower()


def test_p4_file_type_allowlist_rejects_exe_with_415(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_test_models: tuple[Path, Path],
) -> None:
    """P4: Uploading a Windows executable (MZ...) or disguised binary returns HTTP 415."""
    hybrid_dir, crf_dir = isolated_test_models
    monkeypatch.setenv("MEDRESEARCH_DB_PATH", str(tmp_path / "p4.sqlite3"))
    monkeypatch.setenv("MEDRESEARCH_SQLITE_PATH", str(tmp_path / "p4.sqlite3"))
    monkeypatch.setenv("MEDRESEARCH_STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("MEDRESEARCH_MODEL_DIR", str(hybrid_dir))
    monkeypatch.setenv("MEDRESEARCH_CLASSICAL_CRF_DIR", str(crf_dir))
    monkeypatch.setenv("MEDRESEARCH_DEMO_PASSWORD", "ChangeMe_LocalDev_2026!")

    app = create_app(PlatformConfig.from_env())
    client = TestClient(app)
    headers = _get_auth_headers(client)

    mz_bytes = b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff\x00\x00This program cannot be run in DOS mode."
    resp_exe = client.post(
        "/api/v1/documents/upload",
        headers=headers,
        files={"file": ("program.exe", mz_bytes, "application/octet-stream")},
    )
    assert resp_exe.status_code == 415

    resp_disguised = client.post(
        "/api/v1/documents/upload",
        headers=headers,
        files={"file": ("disguised.txt", mz_bytes, "text/plain")},
    )
    assert resp_disguised.status_code == 415


def test_p5_size_limits_413_and_300k_char_chunked_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_test_models: tuple[Path, Path],
) -> None:
    """P5: 11 MB file returns 413; 300k-char note succeeds in sentence-window chunks without crashing."""
    hybrid_dir, crf_dir = isolated_test_models
    monkeypatch.setenv("MEDRESEARCH_DB_PATH", str(tmp_path / "p5.sqlite3"))
    monkeypatch.setenv("MEDRESEARCH_SQLITE_PATH", str(tmp_path / "p5.sqlite3"))
    monkeypatch.setenv("MEDRESEARCH_STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("MEDRESEARCH_MODEL_DIR", str(hybrid_dir))
    monkeypatch.setenv("MEDRESEARCH_CLASSICAL_CRF_DIR", str(crf_dir))
    monkeypatch.setenv("MEDRESEARCH_DEMO_PASSWORD", "ChangeMe_LocalDev_2026!")

    app = create_app(PlatformConfig.from_env())
    client = TestClient(app)
    headers = _get_auth_headers(client)

    eleven_mb = b"A" * (11 * 1024 * 1024)
    resp_big = client.post(
        "/api/v1/documents/upload",
        headers=headers,
        files={"file": ("huge.txt", eleven_mb, "text/plain")},
    )
    assert resp_big.status_code == 413

    # 300k-character clinical note
    sentence = "Patient presents with pneumonia and takes amoxicillin 1000 mg twice daily. No fever or chest pain.\n"
    long_note = (sentence * (300_500 // len(sentence) + 1)).encode("utf-8")
    assert len(long_note) >= 300_000

    up = client.post(
        "/api/v1/documents/upload",
        headers=headers,
        files={"file": ("long_300k.txt", long_note, "text/plain")},
    )
    assert up.status_code == 202
    job_id = up.json()["job_id"]

    # Poll until completed while checking /api/v1/health latency (length-scaled timeout for 4-core CPUs)
    deadline = time.perf_counter() + 360.0
    final_job = None
    while time.perf_counter() < deadline:
        t0 = time.perf_counter()
        h = client.get("/api/v1/health")
        assert h.status_code == 200
        assert (time.perf_counter() - t0) < 5.0
        jr = client.get(f"/api/v1/jobs/{job_id}", headers=headers).json()
        if jr["status"] in ("SUCCEEDED", "FAILED"):
            final_job = jr
            break
        time.sleep(0.1)

    assert final_job is not None
    assert final_job["status"] == "SUCCEEDED", f"300k char job failed: {final_job}"
    assert final_job["result"]["entities_extracted"] > 0


def test_p6_document_date_extracted_from_text_and_anchors_relative_timex(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_test_models: tuple[Path, Path],
) -> None:
    """P6: Extract document date from text first (e.g. 14/11/2026) and anchor relative TIMEX on it."""
    hybrid_dir, crf_dir = isolated_test_models
    monkeypatch.setenv("MEDRESEARCH_DB_PATH", str(tmp_path / "p6.sqlite3"))
    monkeypatch.setenv("MEDRESEARCH_SQLITE_PATH", str(tmp_path / "p6.sqlite3"))
    monkeypatch.setenv("MEDRESEARCH_STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("MEDRESEARCH_MODEL_DIR", str(hybrid_dir))
    monkeypatch.setenv("MEDRESEARCH_CLASSICAL_CRF_DIR", str(crf_dir))
    monkeypatch.setenv("MEDRESEARCH_DEMO_PASSWORD", "ChangeMe_LocalDev_2026!")

    app = create_app(PlatformConfig.from_env())
    client = TestClient(app)
    headers = _get_auth_headers(client)

    note = (
        "Date : 14/11/2026\n"
        "Motif : Patiente présentant une dyspnée depuis 3 semaines.\n"
        "Plan : Contrôle cardiologique dans 2 semaines.\n"
    )
    resp = client.post(
        "/api/v1/documents/ingest",
        headers=headers,
        json={
            "filename": "note_date_fr.txt",
            "text": note,
            "dataset_id": "ds_synth_bilingual_v1",
            "subject_id": "subj_p6",
            "family_id": "fam_p6",
            "encounter_date_iso": "2026-06-01",
        },
    )
    assert resp.status_code in (200, 201)
    body = resp.json()
    assert body.get("document_date") == "2026-11-14"
    assert body.get("document_date_source") == "extracted_from_text"

    svc = ExtractionService(model_dir=hybrid_dir)
    ext = svc.extract(note, language="fr", document_date="2026-11-14")
    all_timex = ext.records[0].provenance.get("timex", []) if ext.records else []
    norm_by_text = {t["text"].lower(): t["normalized"] for t in all_timex}
    assert norm_by_text.get("depuis 3 semaines") == "2026-10-24"
    assert norm_by_text.get("dans 2 semaines") == "2026-11-28"


def test_p7_duplicate_upload_returns_existing_document_with_duplicate_of(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_test_models: tuple[Path, Path],
) -> None:
    """P7: Re-uploading the exact same content_sha256 in the same institution returns 200 with duplicate_of."""
    hybrid_dir, crf_dir = isolated_test_models
    monkeypatch.setenv("MEDRESEARCH_DB_PATH", str(tmp_path / "p7.sqlite3"))
    monkeypatch.setenv("MEDRESEARCH_SQLITE_PATH", str(tmp_path / "p7.sqlite3"))
    monkeypatch.setenv("MEDRESEARCH_STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("MEDRESEARCH_MODEL_DIR", str(hybrid_dir))
    monkeypatch.setenv("MEDRESEARCH_CLASSICAL_CRF_DIR", str(crf_dir))
    monkeypatch.setenv("MEDRESEARCH_DEMO_PASSWORD", "ChangeMe_LocalDev_2026!")

    app = create_app(PlatformConfig.from_env())
    client = TestClient(app)
    headers = _get_auth_headers(client)

    note_bytes = b"Date: 2026-11-02\nPatient has hypertension and takes lisinopril 10 mg once daily."
    r1 = client.post(
        "/api/v1/documents/ingest",
        headers=headers,
        json={"filename": "dup1.txt", "text": note_bytes.decode("utf-8"), "subject_id": "subj_dup"},
    )
    assert r1.status_code == 201
    doc1_id = r1.json()["document_id"]

    r2 = client.post(
        "/api/v1/documents/ingest",
        headers=headers,
        json={"filename": "dup2.txt", "text": note_bytes.decode("utf-8"), "subject_id": "subj_dup"},
    )
    assert r2.status_code == 200
    assert r2.json().get("duplicate_of") == doc1_id
    assert r2.json().get("document_id") == doc1_id


def test_p8_all_api_v1_routes_require_bearer_token_401(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_test_models: tuple[Path, Path],
) -> None:
    """P8: Every /api/v1/* route in OpenAPI except /health and /auth/token returns 401 without token."""
    hybrid_dir, crf_dir = isolated_test_models
    monkeypatch.setenv("MEDRESEARCH_DB_PATH", str(tmp_path / "p8.sqlite3"))
    monkeypatch.setenv("MEDRESEARCH_SQLITE_PATH", str(tmp_path / "p8.sqlite3"))
    monkeypatch.setenv("MEDRESEARCH_STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("MEDRESEARCH_MODEL_DIR", str(hybrid_dir))
    monkeypatch.setenv("MEDRESEARCH_CLASSICAL_CRF_DIR", str(crf_dir))
    monkeypatch.setenv("MEDRESEARCH_DEMO_PASSWORD", "ChangeMe_LocalDev_2026!")

    app = create_app(PlatformConfig.from_env())
    client = TestClient(app)
    openapi = app.openapi()
    exempt = {"/api/v1/health", "/api/v1/auth/token"}

    for path, methods in openapi.get("paths", {}).items():
        if not path.startswith("/api/v1/") or path in exempt:
            continue
        concrete_path = (
            path.replace("{document_id}", "doc_test")
            .replace("{job_id}", "job_test")
            .replace("{entity_id}", "ent_test")
            .replace("{subject_id}", "subj_test")
        )
        for method in methods:
            m = method.upper()
            if m not in {"GET", "POST", "PUT", "DELETE", "PATCH"}:
                continue
            resp = client.request(m, concrete_path)
            assert resp.status_code == 401, f"Expected 401 on unauthenticated {m} {path}, got {resp.status_code}"


def test_p9_job_response_has_unified_error_filename_stage_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_test_models: tuple[Path, Path],
) -> None:
    """P9: GET /api/v1/jobs and GET /api/v1/jobs/{id} include error, original_filename, stage, progress_pct."""
    hybrid_dir, crf_dir = isolated_test_models
    monkeypatch.setenv("MEDRESEARCH_DB_PATH", str(tmp_path / "p9.sqlite3"))
    monkeypatch.setenv("MEDRESEARCH_SQLITE_PATH", str(tmp_path / "p9.sqlite3"))
    monkeypatch.setenv("MEDRESEARCH_STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("MEDRESEARCH_MODEL_DIR", str(hybrid_dir))
    monkeypatch.setenv("MEDRESEARCH_CLASSICAL_CRF_DIR", str(crf_dir))
    monkeypatch.setenv("MEDRESEARCH_DEMO_PASSWORD", "ChangeMe_LocalDev_2026!")

    app = create_app(PlatformConfig.from_env())
    client = TestClient(app)
    headers = _get_auth_headers(client)

    up = client.post(
        "/api/v1/documents/upload",
        headers=headers,
        files={"file": ("clinical_note.txt", b"Patient has diabetes and takes metformin 500 mg.", "text/plain")},
    )
    assert up.status_code == 202
    job_id = up.json()["job_id"]

    # Wait briefly if running asynchronously
    for _ in range(50):
        detail = client.get(f"/api/v1/jobs/{job_id}", headers=headers).json()
        if detail["status"] in ("SUCCEEDED", "FAILED"):
            break
        time.sleep(0.05)

    detail = client.get(f"/api/v1/jobs/{job_id}", headers=headers).json()
    for required_key in ("error", "original_filename", "stage", "progress_pct"):
        assert required_key in detail, f"Missing {required_key} in job detail: {detail}"
    assert detail["original_filename"] == "clinical_note.txt"
