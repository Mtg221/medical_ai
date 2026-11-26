"""Step 0 safety tests (V8, V9, V10): fail-loud missing/tampered models (503), strict subject_id (404/422), and no invented capture date."""

from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from medresearch_platform.config import PlatformConfig
from medresearch_platform.interfaces.api import create_app
from medresearch_platform.vision.cnn_multitask import DermResNetMultiTask
from medresearch_platform.vision.router import (
    PictureTriageRouter,
    generate_synthetic_skin_modality_image,
)
from medresearch_platform.vision.serving import VisionServingEngine


def _encode_png(img: Image.Image, *, with_exif_date: str | None = None) -> bytes:
    buf = io.BytesIO()
    if with_exif_date:
        exif = Image.Exif()
        exif[306] = with_exif_date
        img.save(buf, format="PNG", exif=exif.tobytes())
    else:
        img.save(buf, format="PNG")
    return buf.getvalue()


def _create_tiny_trained_models(workspace: Path) -> None:
    router = PictureTriageRouter(seed=42)
    router.save(workspace / "models" / "vision_router_v1")
    skin = DermResNetMultiTask(image_size=64, seed=42)
    skin.save(workspace / "models" / "derm_cnn_seed1")


def _auth_headers(client: TestClient) -> dict[str, str]:
    tok = client.post(
        "/api/v1/auth/token",
        json={"username": "admin_dakar", "password": "TestPass_2026!"},
    ).json()["access_token"]
    return {"Authorization": f"Bearer {tok}"}


@pytest.fixture(autouse=True)
def _isolate_db_and_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEDRESEARCH_SQLITE_PATH", str(tmp_path / "test_step0.sqlite3"))
    monkeypatch.setenv("MEDRESEARCH_STORAGE_DIR", str(tmp_path / "storage"))


def test_v8_missing_and_tampered_models_return_503_and_store_nothing(tmp_path: Path) -> None:
    cfg = PlatformConfig.from_env(workspace_dir=tmp_path)
    app = create_app(cfg)
    app.state.service.init_or_reset_default_users("TestPass_2026!")
    client = TestClient(app)
    headers = _auth_headers(client)

    # 1. /api/v1/health reports missing vision models
    health = client.get("/api/v1/health")
    assert health.status_code == 200
    hdata = health.json()
    assert hdata["vision_router"] == "missing"
    assert hdata["vision_skin"] == "missing"

    # 2. VisionServingEngine raises ModelNotAvailableError
    from medresearch_platform.vision.serving import ModelNotAvailableError

    engine = VisionServingEngine(workspace_dir=tmp_path)
    with pytest.raises(ModelNotAvailableError):
        _ = engine.router
    with pytest.raises(ModelNotAvailableError):
        _ = engine.skin_model

    # 3. Upload fails with 503 and stores nothing
    skin_im = generate_synthetic_skin_modality_image("skin_dermoscopic", 1, seed=42)
    raw_png = _encode_png(skin_im)
    resp = client.post(
        "/api/v1/images/upload",
        headers=headers,
        files={"file": ("lesion.png", raw_png, "image/png")},
        data={"subject_id": "subj_synth_001"},
    )
    assert resp.status_code == 503, resp.text
    body = resp.json()
    assert "image model not trained" in body["error"].lower()
    assert "medextract derm train" in body["command"]

    with app.state.service.db.connect() as conn:
        count = conn.execute("SELECT COUNT(*) FROM image_findings").fetchone()[0]
        assert count == 0
    assert not list((cfg.storage_dir / "originals").glob("*"))

    # 4. Tampered SHA-256 also raises ModelNotAvailableError and returns 503
    _create_tiny_trained_models(tmp_path)
    health_ready = client.get("/api/v1/health").json()
    assert health_ready["vision_router"] == "ready"
    assert health_ready["vision_skin"] == "ready"

    manifest_path = tmp_path / "models" / "vision_router_v1" / "manifest.json"
    mdata = json.loads(manifest_path.read_text(encoding="utf-8"))
    mdata["trained_weights_sha256"] = hashlib.sha256(b"tampered").hexdigest()
    manifest_path.write_text(json.dumps(mdata), encoding="utf-8")

    # Reset cached engine on app.state.service
    app.state.service._vision_engine = VisionServingEngine(workspace_dir=tmp_path)
    resp_tampered = client.post(
        "/api/v1/images/upload",
        headers=headers,
        files={"file": ("lesion.png", raw_png, "image/png")},
        data={"subject_id": "subj_synth_001"},
    )
    assert resp_tampered.status_code == 503


def test_v9_unknown_or_missing_subject_never_falls_back_to_other_patient(tmp_path: Path) -> None:
    _create_tiny_trained_models(tmp_path)
    cfg = PlatformConfig.from_env(workspace_dir=tmp_path)
    app = create_app(cfg)
    app.state.service.init_or_reset_default_users("TestPass_2026!")
    client = TestClient(app)
    headers = _auth_headers(client)

    # Seed two patients in the same institution
    now = datetime.now(timezone.utc).isoformat()
    with app.state.service.db.transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO datasets ("
            "dataset_id, institution_id, name, version, source_type, license, consent_basis, "
            "permitted_purpose, deidentification_status, languages_json, modalities_json, "
            "record_count, manifest_sha256, retention_policy, created_at"
            ") VALUES ('ds_synth_bilingual_v1', 'inst_dakar_primary', 'Test', '3.1', 'SYNTHETIC', 'MIT', "
            "'RESEARCH', 'BENCHMARK', 'SYNTHETIC_NO_PHI', '[\"en\",\"fr\"]', '[\"TXT\"]', 0, 'abc', 'RETAIN', ?)",
            (now,),
        )
        for sid, code in [("subj_patient_a", "PAT-A"), ("subj_patient_b", "PAT-B")]:
            conn.execute(
                "INSERT OR IGNORE INTO subjects (subject_id, institution_id, dataset_id, pseudonym_code, language_primary, consent_status, created_at) "
                "VALUES (?, 'inst_dakar_primary', 'ds_synth_bilingual_v1', ?, 'en', 'SYNTHETIC_EXEMPT', ?)",
                (sid, code, now),
            )

    skin_im = generate_synthetic_skin_modality_image("skin_dermoscopic", 2, seed=42)
    raw_png = _encode_png(skin_im)

    # 1. Missing subject_id -> 422, nothing stored
    resp_missing = client.post(
        "/api/v1/images/upload",
        headers=headers,
        files={"file": ("lesion_missing_subj.png", raw_png, "image/png")},
    )
    assert resp_missing.status_code == 422, resp_missing.text

    # 2. Unknown subject_id -> 404, nothing stored on either patient
    resp_unknown = client.post(
        "/api/v1/images/upload",
        headers=headers,
        files={"file": ("lesion_unknown_subj.png", raw_png, "image/png")},
        data={"subject_id": "subj_does_not_exist_999"},
    )
    assert resp_unknown.status_code == 404, resp_unknown.text

    with app.state.service.db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM image_findings").fetchone()[0] == 0
    assert not list((cfg.storage_dir / "originals").glob("*"))


def test_v10_missing_capture_date_is_null_unknown_and_warned(tmp_path: Path) -> None:
    _create_tiny_trained_models(tmp_path)
    cfg = PlatformConfig.from_env(workspace_dir=tmp_path)
    app = create_app(cfg)
    app.state.service.init_or_reset_default_users("TestPass_2026!")
    client = TestClient(app)
    headers = _auth_headers(client)

    now = datetime.now(timezone.utc).isoformat()
    with app.state.service.db.transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO datasets ("
            "dataset_id, institution_id, name, version, source_type, license, consent_basis, "
            "permitted_purpose, deidentification_status, languages_json, modalities_json, "
            "record_count, manifest_sha256, retention_policy, created_at"
            ") VALUES ('ds_synth_bilingual_v1', 'inst_dakar_primary', 'Test', '3.1', 'SYNTHETIC', 'MIT', "
            "'RESEARCH', 'BENCHMARK', 'SYNTHETIC_NO_PHI', '[\"en\",\"fr\"]', '[\"TXT\"]', 0, 'abc', 'RETAIN', ?)",
            (now,),
        )
        conn.execute(
            "INSERT OR IGNORE INTO subjects (subject_id, institution_id, dataset_id, pseudonym_code, language_primary, consent_status, created_at) "
            "VALUES ('subj_date_test', 'inst_dakar_primary', 'ds_synth_bilingual_v1', 'PAT-DATE', 'en', 'SYNTHETIC_EXEMPT', ?)",
            (now,),
        )

    skin_im = generate_synthetic_skin_modality_image("skin_dermoscopic", 3, seed=42)
    raw_png_no_exif = _encode_png(skin_im, with_exif_date=None)

    resp = client.post(
        "/api/v1/images/upload",
        headers=headers,
        files={"file": ("lesion_nodate.png", raw_png_no_exif, "image/png")},
        data={"subject_id": "subj_date_test", "body_site": "back"},
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["captured_date_iso"] is None
    assert data["captured_date_source"] == "unknown"
    assert any("date missing" in w.lower() for w in data.get("warnings", []))

    # Timeline places it under "date unknown", never today's date
    rec = client.get("/api/v1/patients/subj_date_test/record", headers=headers)
    assert rec.status_code == 200
    skin_events = [e for e in rec.json()["timeline_events"] if e["event_category"] == "SKIN_IMAGE_FINDING"]
    assert len(skin_events) == 1
    assert skin_events[0]["event_date_iso"] is None
    assert skin_events[0]["date_unknown"] is True
