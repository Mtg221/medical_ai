from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw, ImageFont

from medresearch_platform.config import PlatformConfig
from medresearch_platform.interfaces.api import create_app
from medresearch_platform.security.auth import AuthenticatedPrincipal
from medresearch_platform.vision.quality import detect_dense_text_lines
from medresearch_platform.vision.router import TriageDecision
from medresearch_platform.vision.serving import ingest_and_analyze_image


def _make_dense_text_page() -> Image.Image:
    im = Image.new("RGB", (320, 400), (244, 242, 236))
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default()
    lines = [
        "LABORATORY REPORT - PATIENT RECORD",
        "Date: 2026-11-18  Hemoglobin: 13.4 g/dL",
        "WBC: 6.8 x10^9/L  Platelets: 245 x10^9/L",
        "Glucose fasting: 94 mg/dL (Normal)",
        "Creatinine: 0.88 mg/dL  eGFR > 90",
        "Prescription: Amoxicillin 500 mg TID x7d",
        "Follow-up: Dermatology clinic in 2 weeks.",
        "Clinician: Dr. Fatou Diallo, Dakar",
    ]
    y = 24
    for ln in lines * 2:
        draw.text((16, y), ln, fill=(18, 20, 24), font=font)
        y += 20
    return im


def _make_smooth_skin_lesion() -> Image.Image:
    im = Image.new("RGB", (128, 128), (212, 162, 136))
    draw = ImageDraw.Draw(im)
    draw.ellipse([(38, 36), (92, 88)], fill=(120, 72, 54))
    draw.ellipse([(50, 48), (78, 74)], fill=(86, 48, 36))
    return im


def test_detect_dense_text_lines_separates_document_and_skin() -> None:
    doc_im = _make_dense_text_page()
    doc_check = detect_dense_text_lines(doc_im)
    assert doc_check["has_dense_text"] is True
    assert doc_check["text_line_bands"] >= 4

    skin_im = _make_smooth_skin_lesion()
    skin_check = detect_dense_text_lines(skin_im)
    assert skin_check["has_dense_text"] is False


def test_serving_text_guard_blocks_skin_model_even_if_router_misroutes(tmp_path: Path) -> None:
    cfg = PlatformConfig.from_env(workspace_dir=tmp_path)
    app = create_app(cfg)
    svc = app.state.service
    svc.init_or_reset_default_users("TestPass_2026!")
    now_iso = datetime.now(timezone.utc).isoformat()
    with svc.db.transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO datasets ("
            "dataset_id, institution_id, name, version, source_type, license, consent_basis, "
            "permitted_purpose, deidentification_status, languages_json, modalities_json, "
            "record_count, manifest_sha256, retention_policy, created_at"
            ") VALUES ('ds_synth_bilingual_v1', 'inst_dakar_primary', 'Test', '3.1', 'SYNTHETIC', 'MIT', "
            "'RESEARCH', 'BENCHMARK', 'SYNTHETIC_NO_PHI', '[\"en\",\"fr\"]', '[\"TXT\"]', 0, 'abc', 'RETAIN', ?)",
            (now_iso,),
        )
        conn.execute(
            "INSERT INTO subjects (subject_id, institution_id, dataset_id, pseudonym_code, language_primary, consent_status, created_at) "
            "VALUES ('subj_guard_1', 'inst_dakar_primary', 'ds_synth_bilingual_v1', 'SUBJ-GRD1', 'en', 'SYNTHETIC_EXEMPT', ?)",
            (now_iso,),
        )

    class _MisroutingRouter:
        def triage_image(self, _im: Image.Image) -> TriageDecision:
            return TriageDecision(
                predicted_class="skin_clinical",
                route_action="skin_vision_pipeline",
                confidence=0.60,
                probabilities={"skin_clinical": 0.60, "skin_dermoscopic": 0.05, "skin_tbp_tile": 0.05, "document_scan": 0.25, "other": 0.05},
                label_en="Clinical skin photo",
                label_fr="Photo clinique",
                quality=SimpleNamespace(quality_ok=True, to_dict=lambda: {"quality_ok": True}),  # type: ignore[arg-type]
                abstained=False,
                abstain_reason_en=None,
                abstain_reason_fr=None,
                model_name="FakeRouter",
                pretrained_weights_used=False,
            )

    skin_called = {"count": 0}

    class _SentinelSkinModel:
        metadata_mode = "full"
        include_metadata = True
        architecture_name = "SentinelResNet"
        abstain_min_confidence = 0.55

        def predict_image(self, *_args, **_kwargs):
            skin_called["count"] += 1
            raise AssertionError("Skin model must NEVER be called on dense-text document image!")

        def compute_gradcam_heatmap(self, _im: Image.Image):
            import numpy as np

            return np.zeros((8, 8), dtype=np.float64), 0.0

        def render_gradcam_overlay_png(self, im: Image.Image, _cam) -> bytes:
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            return buf.getvalue()

    fake_engine = SimpleNamespace(router=_MisroutingRouter(), skin_model=_SentinelSkinModel())
    doc_im = _make_dense_text_page()
    buf = io.BytesIO()
    doc_im.save(buf, format="PNG")

    principal = AuthenticatedPrincipal(
        user_id="user_admin_dakar",
        username="admin_dakar",
        role="INSTITUTION_ADMIN",
        institution_id="inst_dakar_primary",
        permissions=frozenset({"images:write", "images:read", "medical_data:write"}),
        expires_at_iso="2099-01-01T00:00:00Z",
    )
    res = ingest_and_analyze_image(
        svc,
        principal=principal,
        filename="lab_report.png",
        raw_bytes=buf.getvalue(),
        subject_id="subj_guard_1",
        allow_document_reroute=False,
        vision_engine=fake_engine,
    )
    assert skin_called["count"] == 0
    assert res["picture_type"] == "document_scan"
    assert res["malignancy_risk_probability"] is None
    assert res["malignancy_risk_band"] == "abstained"
    assert res["abstained"] is True
    assert res["provenance"]["text_presence_check"]["has_dense_text"] is True
