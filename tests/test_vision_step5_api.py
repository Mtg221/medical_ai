"""Step 5 tests: FastAPI image upload (EXIF/GPS stripping), bidirectional OCR/skin routing, heatmap & patient record endpoints, EN/FR reports, and institution isolation."""

from __future__ import annotations

import io
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from medresearch_platform.config import PlatformConfig
from medresearch_platform.interfaces.api import create_app
from medresearch_platform.vision.router import (
    generate_document_scan_image,
    generate_synthetic_skin_modality_image,
)


def _encode_png_with_exif(img: Image.Image) -> bytes:
    exif = Image.Exif()
    exif[306] = "2026:11:15 14:22:00"  # DateTime
    exif[271] = "TestCameraMaker-GPS-Leak"
    buf = io.BytesIO()
    img.save(buf, format="PNG", exif=exif.tobytes())
    return buf.getvalue()


def test_step5_image_upload_exif_stripping_linking_and_endpoints(tmp_path: Path) -> None:
    """Verify /api/v1/images/upload, EXIF stripping, SAME_LESION linking, heatmap, review, patient record, and EN/FR reports."""
    cfg = PlatformConfig.from_env(workspace_dir=tmp_path)
    app = create_app(cfg)
    app.state.service.init_or_reset_default_users("TestPass_2026!")
    client = TestClient(app)

    # 1. Authenticate as admin_dakar (inst_dakar_primary) and researcher_paris (inst_paris_partner)
    tok_dakar = client.post(
        "/api/v1/auth/token",
        json={"username": "admin_dakar", "password": "TestPass_2026!"},
    ).json()["access_token"]
    tok_paris = client.post(
        "/api/v1/auth/token",
        json={"username": "researcher_paris", "password": "TestPass_2026!"},
    ).json()["access_token"]

    headers_dakar = {"Authorization": f"Bearer {tok_dakar}"}
    headers_paris = {"Authorization": f"Bearer {tok_paris}"}

    # 2. Ingest a clinical note mentioning a pigmented lesion on the left forearm
    doc_resp = client.post(
        "/api/v1/documents/ingest",
        headers=headers_dakar,
        json={
            "filename": "consult_derm_fr.txt",
            "text": (
                "Consultation de dermatologie du 2026-11-15.\n"
                "Motif : lésion pigmentée de l'avant-bras gauche de 7 mm avec changement récent."
            ),
            "dataset_id": "ds_synth_bilingual_v1",
            "subject_id": "subj_synth_001",
            "family_id": "fam_derm_1",
            "encounter_date_iso": "2026-11-15",
        },
    )
    assert doc_resp.status_code in (200, 201)
    doc_id = doc_resp.json()["document_id"]

    # 3. Upload a dermoscopic skin image with EXIF tags and body_site="left forearm"
    skin_im = generate_synthetic_skin_modality_image("skin_dermoscopic", 1, seed=42)
    raw_png = _encode_png_with_exif(skin_im)

    up_resp = client.post(
        "/api/v1/images/upload",
        headers=headers_dakar,
        files={"file": ("lesion_forearm.png", raw_png, "image/png")},
        data={
            "subject_id": "subj_synth_001",
            "document_id": doc_id,
            "body_site": "left forearm",
        },
    )
    assert up_resp.status_code == 201, up_resp.text
    up_data = up_resp.json()
    assert up_data["stages"] == ["TRIAGE", "QUALITY", "ANALYSE", "EXPLAIN"]
    assert up_data["exif_stripped"] is True
    assert up_data["captured_date_iso"] == "2026-11-15"
    assert up_data["routed_to"] == "skin_vision_pipeline"
    assert len(up_data["same_lesion_links"]) >= 1
    assert up_data["same_lesion_links"][0]["matched_body_site"] == "upper extremity"

    image_id = up_data["image_id"]

    # Verify stored file on disk has zero EXIF metadata
    stored_files = list((cfg.storage_dir / "originals").glob(f"{image_id}_*"))
    assert len(stored_files) == 1
    with Image.open(stored_files[0]) as stored_im:
        assert len(stored_im.getexif()) == 0

    # 4. Duplicate upload detection
    dup_resp = client.post(
        "/api/v1/images/upload",
        headers=headers_dakar,
        files={"file": ("lesion_forearm_copy.png", raw_png, "image/png")},
        data={"subject_id": "subj_synth_001"},
    )
    assert dup_resp.status_code == 201
    assert dup_resp.json()["is_duplicate"] is True
    assert dup_resp.json()["duplicate_of"] == image_id

    # 5. GET /api/v1/images/{id} and GET /api/v1/images/{id}/heatmap
    get_img = client.get(f"/api/v1/images/{image_id}", headers=headers_dakar)
    assert get_img.status_code == 200
    assert get_img.json()["image_id"] == image_id
    assert "thumbnail_base64" in get_img.json()

    hm_resp = client.get(f"/api/v1/images/{image_id}/heatmap", headers=headers_dakar)
    assert hm_resp.status_code == 200
    assert hm_resp.headers["content-type"] == "image/png"
    assert hm_resp.content[:8] == b"\x89PNG\r\n\x1a\n"

    # Cross-institution isolation: Paris user cannot read Dakar image finding or heatmap
    assert client.get(f"/api/v1/images/{image_id}", headers=headers_paris).status_code == 403
    assert client.get(f"/api/v1/images/{image_id}/heatmap", headers=headers_paris).status_code == 403

    # 6. Review persistence on image finding
    rev_resp = client.post(
        f"/api/v1/images/{image_id}/review",
        headers=headers_dakar,
        json={
            "review_decision": "CORRECT_AND_APPROVE",
            "corrected_diagnosis": "melanoma",
            "corrected_risk_band": "high",
            "rationale": "Confirmed by dermatoscope review",
        },
    )
    assert rev_resp.status_code == 200
    assert rev_resp.json()["review_status"] == "corrected"

    # 7. GET /api/v1/patients/{id}/record
    rec_resp = client.get("/api/v1/patients/subj_synth_001/record", headers=headers_dakar)
    assert rec_resp.status_code == 200
    rec_data = rec_resp.json()
    assert len(rec_data["image_findings"]) >= 1
    assert len(rec_data["same_lesion_links"]) >= 1
    assert any(e["event_category"] == "SKIN_IMAGE_FINDING" for e in rec_data["timeline_events"])

    # 8. EN and FR reports contain the Skin Images / Images cutanées section and safety notice
    rep_en = client.get(f"/api/v1/documents/{doc_id}/report?lang=en&format=markdown", headers=headers_dakar)
    assert rep_en.status_code == 200
    assert "## Skin Images" in rep_en.text
    assert "Research tool — not a diagnosis. A dermatologist must examine the lesion." in rep_en.text

    rep_fr = client.get(f"/api/v1/documents/{doc_id}/report?lang=fr&format=markdown", headers=headers_dakar)
    assert rep_fr.status_code == 200
    assert "## Images cutanées" in rep_fr.text
    assert "Outil de recherche — ne constitue pas un diagnostic. Un dermatologue doit examiner la lésion." in rep_fr.text

    # 9. Bidirectional routing: document scan uploaded to /api/v1/images/upload -> routed to OCR
    from medresearch_platform.documents.ocr_engine import check_tesseract_available

    tess_ok, _ = check_tesseract_available()
    doc_scan_im = generate_document_scan_image(5, seed=42)
    doc_buf = io.BytesIO()
    doc_scan_im.save(doc_buf, format="PNG")
    reroute_resp = client.post(
        "/api/v1/images/upload",
        headers=headers_dakar,
        files={"file": ("clinical_form_scan.png", doc_buf.getvalue(), "image/png")},
        data={"subject_id": "subj_synth_001", "force_new": "true"},
    )
    if tess_ok:
        assert reroute_resp.status_code == 201
        assert reroute_resp.json()["routed_to"] == "ocr_text_pipeline"
    else:
        assert reroute_resp.status_code == 400
        assert "Tesseract OCR" in reroute_resp.json()["error"]

    # 10. Vision benchmark results endpoint
    bench_resp = client.get("/api/v1/results/vision/summary", headers=headers_dakar)
    assert bench_resp.status_code == 200
    assert "router_evaluation" in bench_resp.json()
    assert "skin_benchmark" in bench_resp.json()
