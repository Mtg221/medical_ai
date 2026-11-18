from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from medresearch_platform.config import PlatformConfig
from medresearch_platform.security.auth import AuthenticatedPrincipal
from medresearch_platform.vision.catalog import DermCatalogRecord
from medresearch_platform.vision.cnn_multitask import SkinPredictionResult
from medresearch_platform.vision.cnn_torch import encode_metadata_vector
from medresearch_platform.vision.router import TriageDecision
from medresearch_platform.vision.serving import ingest_and_analyze_image


def _make_record(modality: str = "dermoscopic", body_site: str | None = "left forearm") -> DermCatalogRecord:
    return DermCatalogRecord(
        image_id="img_1",
        dataset="isic",
        modality=modality,
        thumb_path=None,
        raw_path=None,
        skin_tone="III",
        diagnosis_raw="nevus",
        malignancy_group="benign",
        body_site=body_site,
        license_name="CC0",
        quality_ok=True,
        confirmation="histopathology",
        group_key="grp_1",
        split_assignment="val",
    )


def test_encode_metadata_vector_modes() -> None:
    rec = _make_record(modality="dermoscopic", body_site="left forearm")
    v_pix = encode_metadata_vector(rec, metadata_mode="pixels_only")
    assert float(np.sum(np.abs(v_pix))) == 0.0

    v_mod = encode_metadata_vector(rec, metadata_mode="modality_only")
    assert float(v_mod[1]) == 1.0  # dermoscopic
    assert float(np.sum(np.abs(v_mod[3:]))) == 0.0

    v_full = encode_metadata_vector(rec, metadata_mode="full")
    assert float(v_full[1]) == 1.0  # dermoscopic
    assert float(v_full[5]) == 1.0  # upper extremity (forearm)


def test_serving_passes_modality_hint_and_body_site(tmp_path: Path, monkeypatch) -> None:
    from datetime import datetime, timezone

    from medresearch_platform.interfaces.api import create_app

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
            "VALUES ('subj_test_meta', 'inst_dakar_primary', 'ds_synth_bilingual_v1', 'SUBJ-META', 'en', 'SYNTHETIC_EXEMPT', ?)",
            (now_iso,),
        )

    captured: dict[str, object] = {}

    class _FakeRouter:
        def triage_image(self, _im: Image.Image) -> TriageDecision:
            return TriageDecision(
                predicted_class="skin_dermoscopic",
                route_action="skin_vision_pipeline",
                confidence=0.92,
                probabilities={"skin_dermoscopic": 0.92, "skin_clinical": 0.05, "skin_tbp_tile": 0.01, "document_scan": 0.01, "other": 0.01},
                label_en="Dermoscopic skin photo",
                label_fr="Photo dermoscopique",
                quality=SimpleNamespace(quality_ok=True, to_dict=lambda: {"quality_ok": True}),  # type: ignore[arg-type]
                abstained=False,
                abstain_reason_en=None,
                abstain_reason_fr=None,
                model_name="FakeRouter",
                pretrained_weights_used=False,
            )

    class _FakeSkinModel:
        metadata_mode = "full"
        include_metadata = True
        architecture_name = "FakeResNet"
        abstain_min_confidence = 0.55

        def predict_image(
            self,
            _im: Image.Image,
            *,
            modality_hint: str | None = None,
            body_site: str | None = None,
        ) -> SkinPredictionResult:
            captured["modality_hint"] = modality_hint
            captured["body_site"] = body_site
            return SkinPredictionResult(
                malignancy_probability=0.18,
                risk_band="low",
                top3_diagnoses=[{"rank": 1, "diagnosis": "nevus", "probability": 0.72}],
                predicted_modality=modality_hint or "clinical",
                modality_probabilities={"clinical": 0.05, "dermoscopic": 0.94, "tbp_tile": 0.01},
                energy_score=-1.0,
                confidence=0.82,
                abstained=False,
                abstain_reason_en=None,
                abstain_reason_fr=None,
                needs_review=False,
                quality=SimpleNamespace(quality_ok=True, to_dict=lambda: {"quality_ok": True}),  # type: ignore[arg-type]
                heatmap_grid=[[0.0] * 8 for _ in range(8)],
                safety_notice_en="notice",
                safety_notice_fr="avis",
                model_name="FakeResNet",
                pretrained_weights_used=False,
            )

        def render_gradcam_overlay_png(self, im: Image.Image, _cam: np.ndarray) -> bytes:
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            return buf.getvalue()

    fake_engine = SimpleNamespace(
        router=_FakeRouter(),
        skin_model=_FakeSkinModel(),
    )

    im = Image.new("RGB", (96, 96), (210, 160, 140))
    buf = io.BytesIO()
    im.save(buf, format="PNG")

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
        filename="lesion.png",
        raw_bytes=buf.getvalue(),
        subject_id="subj_test_meta",
        body_site="left forearm",
        vision_engine=fake_engine,
    )
    assert captured["modality_hint"] == "dermoscopic"
    assert captured["body_site"] == "left forearm"
    prov = res["provenance"]
    assert prov["metadata_mode"] == "full"
    assert prov["modality_hint_used"] == "dermoscopic"
    assert prov["body_site_used"] == "left forearm"
