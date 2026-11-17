"""Automated End-to-End Acceptance Test Suite (Text Part D + Real CC0 Images, Privacy, Auth, 404/503, and EN/FR Reports)."""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image, ImageFilter

from medresearch_platform.config import PlatformConfig
from medresearch_platform.documents.ocr_engine import check_tesseract_available
from medresearch_platform.interfaces.api import create_app
from medresearch_platform.security.audit import verify_audit_chain
from medresearch_platform.vision.cnn_multitask import DermResNetMultiTask
from medresearch_platform.vision.router import (
    ROUTER_CLASSES,
    PictureTriageRouter,
    extract_router_features,
)

VISION_ACC_DIR = Path("tests/data/vision_acceptance")


def _encode_with_exif(img_path: Path, *, with_exif: bool = False) -> bytes:
    with Image.open(img_path) as im:
        rgb = im.convert("RGB")
        buf = io.BytesIO()
        if with_exif:
            exif = Image.Exif()
            exif[306] = "2026:11:16 10:30:00"
            exif[271] = "AcceptanceTestCamera-GPS"
            rgb.save(buf, format="PNG", exif=exif.tobytes())
        else:
            rgb.save(buf, format="PNG")
        return buf.getvalue()


def _train_tiny_cc0_models(workspace: Path) -> None:
    """Train tiny router & skin model on the committed CC0 acceptance set for CI smoke runs."""
    skin_paths = sorted((VISION_ACC_DIR / "skin").glob("*"))
    doc_paths = sorted((VISION_ACC_DIR / "documents").glob("*"))
    other_paths = sorted((VISION_ACC_DIR / "non_skin").glob("*"))

    router = PictureTriageRouter(seed=42)
    samples = []
    for sp in skin_paths:
        with Image.open(sp) as im:
            p24, d28 = extract_router_features(im.convert("RGB"))
            samples.append((p24, d28, ROUTER_CLASSES.index("skin_dermoscopic")))
    for dp in doc_paths:
        with Image.open(dp) as im:
            p24, d28 = extract_router_features(im.convert("RGB"))
            samples.append((p24, d28, ROUTER_CLASSES.index("document_scan")))
    for op in other_paths:
        with Image.open(op) as im:
            p24, d28 = extract_router_features(im.convert("RGB"))
            samples.append((p24, d28, ROUTER_CLASSES.index("other")))
    router.fit(samples, epochs=40, lr=0.06)  # type: ignore[arg-type]
    router.save(workspace / "models" / "vision_router_v1")

    DermResNetMultiTask(image_size=64, seed=42).save(workspace / "models" / "derm_cnn_seed1")


def run_acceptance_suite(*, use_repo_models: bool = False) -> dict[str, object]:
    t0 = time.perf_counter()
    results: dict[str, bool] = {}

    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        old_sqlite = os.environ.get("MEDRESEARCH_SQLITE_PATH")
        old_storage = os.environ.get("MEDRESEARCH_STORAGE_DIR")
        os.environ["MEDRESEARCH_SQLITE_PATH"] = str(tmp_path / "acc.sqlite3")
        os.environ["MEDRESEARCH_STORAGE_DIR"] = str(tmp_path / "storage")

        try:
            # 0. Check no-model -> 503 before installing models
            cfg = PlatformConfig.from_env(workspace_dir=tmp_path)
            app = create_app(cfg)
            app.state.service.init_or_reset_default_users("AcceptancePass_2026!")
            now_iso = datetime.now(timezone.utc).isoformat()
            with app.state.service.db.transaction() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO datasets ("
                    "dataset_id, institution_id, name, version, source_type, license, consent_basis, "
                    "permitted_purpose, deidentification_status, languages_json, modalities_json, "
                    "record_count, manifest_sha256, retention_policy, created_at"
                    ") VALUES ('ds_synth_bilingual_v1', 'inst_dakar_primary', 'Acceptance', '3.1', 'SYNTHETIC', 'MIT', "
                    "'RESEARCH', 'BENCHMARK', 'SYNTHETIC_NO_PHI', '[\"en\",\"fr\"]', '[\"TXT\"]', 0, 'abc', 'RETAIN', ?)",
                    (now_iso,),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO subjects (subject_id, institution_id, dataset_id, pseudonym_code, language_primary, consent_status, created_at) "
                    "VALUES ('subj_synth_001', 'inst_dakar_primary', 'ds_synth_bilingual_v1', 'PAT-ACC-001', 'fr', 'SYNTHETIC_EXEMPT', ?)",
                    (now_iso,),
                )

            client = TestClient(app)

            # 1. Auth & tokens (Dakar vs Paris)
            tok_dakar = client.post(
                "/api/v1/auth/token",
                json={"username": "admin_dakar", "password": "AcceptancePass_2026!"},
            ).json()["access_token"]
            tok_paris = client.post(
                "/api/v1/auth/token",
                json={"username": "researcher_paris", "password": "AcceptancePass_2026!"},
            ).json()["access_token"]
            h_dakar = {"Authorization": f"Bearer {tok_dakar}"}
            h_paris = {"Authorization": f"Bearer {tok_paris}"}

            assert client.get("/api/v1/images").status_code == 401
            results["auth_required_on_image_routes"] = True

            skin_files = sorted((VISION_ACC_DIR / "skin").glob("*"))
            doc_files = sorted((VISION_ACC_DIR / "documents").glob("*"))
            other_files = sorted((VISION_ACC_DIR / "non_skin").glob("*"))
            assert len(skin_files) == 5 and len(doc_files) == 5 and len(other_files) == 5

            # No model -> 503
            r_503 = client.post(
                "/api/v1/images/upload",
                headers=h_dakar,
                files={"file": ("skin0.png", _encode_with_exif(skin_files[0]), "image/png")},
                data={"subject_id": "subj_synth_001"},
            )
            assert r_503.status_code == 503
            results["no_model_returns_503"] = True

            # Load repo models (local full-model run) or train tiny CC0 models (CI)
            if use_repo_models and (Path("models/vision_router_v1/manifest.json").exists()):
                shutil.copytree("models/vision_router_v1", tmp_path / "models" / "vision_router_v1")
                shutil.copytree("models/derm_cnn_seed1", tmp_path / "models" / "derm_cnn_seed1")
            else:
                _train_tiny_cc0_models(tmp_path)

            from medresearch_platform.vision.serving import VisionServingEngine

            app.state.service.vision_engine = VisionServingEngine(workspace_dir=tmp_path)

            # Unknown patient -> 404
            r_404 = client.post(
                "/api/v1/images/upload",
                headers=h_dakar,
                files={"file": ("skin0.png", _encode_with_exif(skin_files[0]), "image/png")},
                data={"subject_id": "subj_does_not_exist_999"},
            )
            assert r_404.status_code == 404
            results["unknown_patient_returns_404"] = True

            # 2. Ingest bilingual text note for SAME_LESION linking
            doc_resp = client.post(
                "/api/v1/documents/ingest",
                headers=h_dakar,
                json={
                    "filename": "derm_consult.txt",
                    "text": "Consultation: pigmented lesion on the left forearm measuring 6 mm with recent change.",
                    "dataset_id": "ds_synth_bilingual_v1",
                    "subject_id": "subj_synth_001",
                    "family_id": "fam_acc_1",
                    "encounter_date_iso": "2026-11-16",
                },
            )
            assert doc_resp.status_code in (200, 201)
            doc_id = doc_resp.json()["document_id"]

            # 3. Real CC0 skin image with EXIF/GPS -> triage + Grad-CAM + zero EXIF stored
            up_skin = client.post(
                "/api/v1/images/upload",
                headers=h_dakar,
                files={"file": ("skin_clean.png", _encode_with_exif(skin_files[0], with_exif=True), "image/png")},
                data={"subject_id": "subj_synth_001", "document_id": doc_id, "body_site": "left forearm"},
            )
            assert up_skin.status_code == 201, up_skin.text
            skin_data = up_skin.json()
            image_id = skin_data["image_id"]
            assert skin_data["routed_to"] == "skin_vision_pipeline"
            assert skin_data["exif_stripped"] is True
            results["triage_skin_image"] = True

            stored_files = list((cfg.storage_dir / "originals").glob(f"{image_id}_*"))
            assert len(stored_files) == 1
            with Image.open(stored_files[0]) as stored_im:
                assert len(stored_im.getexif()) == 0
            results["no_exif_gps_kept"] = True

            hm_resp = client.get(f"/api/v1/images/{image_id}/heatmap", headers=h_dakar)
            assert hm_resp.status_code == 200 and hm_resp.content[:8] == b"\x89PNG\r\n\x1a\n"
            results["heatmap_generated"] = True

            # Cross-institution isolation
            assert client.get(f"/api/v1/images/{image_id}", headers=h_paris).status_code == 403
            assert client.get(f"/api/v1/images/{image_id}/heatmap", headers=h_paris).status_code == 403
            results["institution_isolation_on_image_routes"] = True

            # 4. Blurred real skin image -> quality rejection & abstention
            with Image.open(skin_files[1]) as im_s1:
                blur_im = im_s1.convert("RGB").filter(ImageFilter.GaussianBlur(radius=6.0))
                b_buf = io.BytesIO()
                blur_im.save(b_buf, format="PNG")
            up_blur = client.post(
                "/api/v1/images/upload",
                headers=h_dakar,
                files={"file": ("skin_blur.png", b_buf.getvalue(), "image/png")},
                data={"subject_id": "subj_synth_001", "force_new": "true"},
            )
            assert up_blur.status_code == 201
            assert up_blur.json()["abstained"] is True
            assert up_blur.json()["malignancy_risk_band"] == "abstained"
            results["quality_rejection_blurred_image"] = True

            # 5. Real CC0 non-skin image -> not skin -> no risk score
            up_other = client.post(
                "/api/v1/images/upload",
                headers=h_dakar,
                files={"file": ("non_skin.png", _encode_with_exif(other_files[0]), "image/png")},
                data={"subject_id": "subj_synth_001", "force_new": "true"},
            )
            assert up_other.status_code == 201
            assert up_other.json()["abstained"] is True
            assert up_other.json()["malignancy_risk_probability"] is None
            results["non_skin_no_risk_score"] = True

            # 6. Real document photo -> routed to OCR (never sent to skin model)
            up_doc_img = client.post(
                "/api/v1/images/upload",
                headers=h_dakar,
                files={"file": ("doc_scan.png", _encode_with_exif(doc_files[0]), "image/png")},
                data={"subject_id": "subj_synth_001", "force_new": "true"},
            )
            tess_ok, _ = check_tesseract_available()
            if tess_ok:
                assert up_doc_img.status_code == 201 and up_doc_img.json()["routed_to"] == "ocr_text_pipeline"
            else:
                assert up_doc_img.status_code == 400 and "Tesseract OCR" in up_doc_img.json()["error"]
            results["triage_document_photo_to_ocr"] = True

            # 7. Review persistence & audit chain
            rev_resp = client.post(
                f"/api/v1/images/{image_id}/review",
                headers=h_dakar,
                json={"review_decision": "APPROVE_AS_GROUND_TRUTH", "rationale": "Acceptance check"},
            )
            assert rev_resp.status_code == 200 and rev_resp.json()["review_status"] == "approved"
            with app.state.service.db.connect() as conn:
                assert verify_audit_chain(conn)["valid"] is True
            results["review_persistence_and_audit"] = True

            # 8. EN & FR report sections
            rep_en = client.get(f"/api/v1/documents/{doc_id}/report?lang=en&format=markdown", headers=h_dakar)
            rep_fr = client.get(f"/api/v1/documents/{doc_id}/report?lang=fr&format=markdown", headers=h_dakar)
            assert rep_en.status_code == 200 and "## Skin Images" in rep_en.text
            assert rep_fr.status_code == 200 and "## Images cutanées" in rep_fr.text
            results["report_skin_section_en_fr"] = True

            # 9. Part D Text Acceptance Test: two bilingual clinical fixture notes (FR + EN) & P10 redaction
            fr_note_resp = client.post(
                "/api/v1/documents/ingest",
                headers=h_dakar,
                json={
                    "filename": "fixture_fr_bpco.txt",
                    "text": "Mme D, 67 ans, BPCO sous tiotropium 18 µg.",
                    "dataset_id": "ds_synth_bilingual_v1",
                    "subject_id": "subj_synth_001",
                    "family_id": "fam_acc_text",
                    "encounter_date_iso": "2026-11-16",
                },
            )
            assert fr_note_resp.status_code in (200, 201), fr_note_resp.text
            fr_doc_id = fr_note_resp.json()["document_id"]
            fr_detail = client.get(f"/api/v1/documents/{fr_doc_id}", headers=h_dakar).json()
            fr_ents = fr_detail["entities"]
            assert any(e["entity_type"] == "MEDICATION" and "tiotropium" in e["source_text"].lower() for e in fr_ents)
            assert any(e["entity_type"] == "DOSAGE" and "18" in e["source_text"] for e in fr_ents)
            assert not any(e["entity_type"] == "DIAGNOSIS" and "mme" in e["source_text"].lower() for e in fr_ents)
            assert not any(e["entity_type"] == "DATE" and "67 ans" in e["source_text"].lower() for e in fr_ents)
            fr_rep = client.get(f"/api/v1/documents/{fr_doc_id}/report?lang=fr&format=markdown", headers=h_dakar)
            assert fr_rep.status_code == 200 and "Mme D" not in fr_rep.text
            results["text_acceptance_fr_fixture_and_p10"] = True

            en_note_resp = client.post(
                "/api/v1/documents/ingest",
                headers=h_dakar,
                json={
                    "filename": "fixture_en_aspirin.txt",
                    "text": "Patient Name: John Doe. Patient is not taking aspirin.",
                    "dataset_id": "ds_synth_bilingual_v1",
                    "subject_id": "subj_synth_001",
                    "family_id": "fam_acc_text",
                    "encounter_date_iso": "2026-11-16",
                },
            )
            assert en_note_resp.status_code in (200, 201), en_note_resp.text
            en_doc_id = en_note_resp.json()["document_id"]
            en_detail = client.get(f"/api/v1/documents/{en_doc_id}", headers=h_dakar).json()
            en_ents = en_detail["entities"]
            asp = [e for e in en_ents if e["entity_type"] == "MEDICATION" and "aspirin" in e["source_text"].lower()]
            assert len(asp) == 1 and asp[0]["assertion_status"] == "NEGATED"
            en_rep = client.get(f"/api/v1/documents/{en_doc_id}/report?lang=en&format=markdown", headers=h_dakar)
            assert en_rep.status_code == 200 and "John Doe" not in en_rep.text
            results["text_acceptance_en_fixture_and_p10"] = True
        finally:
            if old_sqlite is None:
                os.environ.pop("MEDRESEARCH_SQLITE_PATH", None)
            else:
                os.environ["MEDRESEARCH_SQLITE_PATH"] = old_sqlite
            if old_storage is None:
                os.environ.pop("MEDRESEARCH_STORAGE_DIR", None)
            else:
                os.environ["MEDRESEARCH_STORAGE_DIR"] = old_storage

    return {
        "status": "PASSED",
        "checks": results,
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
    }


if __name__ == "__main__":
    import sys

    use_repo = "--use-repo-models" in sys.argv
    print(json.dumps(run_acceptance_suite(use_repo_models=use_repo), indent=2))
