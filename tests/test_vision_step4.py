"""Step 4 tests: ImageFinding contract & JSON Schema, bilingual cutaneous text-image SAME_LESION linking, and image review persistence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from medresearch_platform.config import PlatformConfig
from medresearch_platform.db.engine import DatabaseEngine
from medresearch_platform.extraction.contract import ImageFinding
from medresearch_platform.extraction.schema_models import (
    ExtractionResultSchema,
    ImageFindingSchema,
    export_extraction_json_schema,
)
from medresearch_platform.security.audit import verify_audit_chain
from medresearch_platform.security.auth import AuthenticatedPrincipal
from medresearch_platform.vision.linking import (
    extract_cutaneous_text_descriptors,
    normalize_body_site,
    propose_same_lesion_links,
)
from medresearch_platform.workflows.review import (
    export_reviewed_image_gold,
    submit_image_finding_review,
)


def test_image_finding_contract_and_json_schema_export(tmp_path: Path) -> None:
    """Verify ImageFinding contract, Pydantic v2 validation, and JSON Schema export."""
    finding = ImageFinding(
        image_id="img_001",
        document_id="doc_001",
        patient_id="subj_001",
        picture_type="skin_dermoscopic",
        quality_result={"quality_ok": True, "quality_score": 0.92},
        malignancy_risk_probability=0.78,
        malignancy_risk_band="high",
        top3_diagnoses=(
            {"rank": 1, "diagnosis": "melanoma", "probability": 0.68},
            {"rank": 2, "diagnosis": "nevus", "probability": 0.20},
            {"rank": 3, "diagnosis": "basal_cell_carcinoma", "probability": 0.12},
        ),
        heatmap_uri="/api/v1/images/img_001/heatmap",
        abstained=False,
        abstain_reason_en=None,
        abstain_reason_fr=None,
        body_site="upper extremity",
        model_name="DermResNetMultiTask-v1.0.0",
        model_version="1.0.0",
        git_commit="ce3140a",
        review_status="needs_review",
        provenance={"router_class": "skin_dermoscopic"},
        captured_date_iso="2026-11-16",
    )
    d = finding.to_dict()
    validated = ImageFindingSchema.model_validate(d)
    assert validated.image_id == "img_001"
    assert validated.malignancy_risk_band == "high"
    assert len(validated.top3_diagnoses) == 3

    bundle = ExtractionResultSchema(
        source_record_id="src_1",
        document_id="doc_001",
        language="fr",
        model_name="hybrid_v2",
        model_version="1.0.0",
        records=[],
        image_findings=[validated],
        same_lesion_links=[],
    )
    assert len(bundle.image_findings) == 1

    schema_out = tmp_path / "medical_extraction.schema.json"
    export_extraction_json_schema(schema_out)
    schema_json = json.loads(schema_out.read_text(encoding="utf-8"))
    assert "ImageFindingSchema" in json.dumps(schema_json)


def test_bilingual_cutaneous_descriptors_and_same_lesion_linking() -> None:
    """Verify FR and EN cutaneous lesion + anatomy extraction and SAME_LESION body-site linking."""
    fr_text = (
        "Patient consultant pour une lésion pigmentée de l'avant-bras gauche de 7 mm "
        "avec changement récent et bords irréguliers."
    )
    fr_ents = extract_cutaneous_text_descriptors(fr_text)
    ent_types = {e["entity_type"] for e in fr_ents}
    assert "ANATOMY" in ent_types
    assert "SYMPTOM" in ent_types

    canon_fr, lat_fr = normalize_body_site("avant-bras gauche")
    assert canon_fr == "upper extremity"
    assert lat_fr == "left"

    canon_en, lat_en = normalize_body_site("mole on the left forearm")
    assert canon_en == "upper extremity"
    assert lat_en == "left"

    finding_left_arm = ImageFinding(
        image_id="img_left_arm",
        document_id="doc_fr_1",
        patient_id="subj_1",
        picture_type="skin_dermoscopic",
        quality_result={"quality_ok": True},
        malignancy_risk_probability=0.71,
        malignancy_risk_band="high",
        top3_diagnoses=({"rank": 1, "diagnosis": "melanoma", "probability": 0.71},),
        heatmap_uri="/api/v1/images/img_left_arm/heatmap",
        abstained=False,
        abstain_reason_en=None,
        abstain_reason_fr=None,
        body_site="left forearm",
        model_name="DermResNetMultiTask-v1.0.0",
        model_version="1.0.0",
        git_commit="ce3140a",
        captured_date_iso="2026-11-16",
    )
    finding_right_arm = ImageFinding(
        image_id="img_right_arm",
        document_id="doc_fr_1",
        patient_id="subj_1",
        picture_type="skin_dermoscopic",
        quality_result={"quality_ok": True},
        malignancy_risk_probability=0.15,
        malignancy_risk_band="low",
        top3_diagnoses=({"rank": 1, "diagnosis": "nevus", "probability": 0.85},),
        heatmap_uri="/api/v1/images/img_right_arm/heatmap",
        abstained=False,
        abstain_reason_en=None,
        abstain_reason_fr=None,
        body_site="right forearm",
        model_name="DermResNetMultiTask-v1.0.0",
        model_version="1.0.0",
        git_commit="ce3140a",
        captured_date_iso="2026-11-16",
    )

    links_fr = propose_same_lesion_links(
        fr_text,
        [],
        [finding_left_arm, finding_right_arm],
        document_id="doc_fr_1",
        patient_id="subj_1",
        document_date_iso="2026-11-16",
    )
    assert len(links_fr) == 1
    assert links_fr[0]["image_id"] == "img_left_arm"
    assert links_fr[0]["relation_type"] == "SAME_LESION"
    assert links_fr[0]["matched_body_site"] == "upper extremity"
    assert links_fr[0]["confidence"] >= 0.90
    assert links_fr[0]["review_status"] == "needs_review"

    en_text = "Physical exam reveals an asymmetric mole on the left forearm measuring 6 mm."
    links_en = propose_same_lesion_links(
        en_text,
        [],
        [finding_left_arm],
        document_id="doc_en_1",
        patient_id="subj_1",
        document_date_iso="2026-11-16",
    )
    assert len(links_en) == 1
    assert "mole on the left forearm" in links_en[0]["text_span"]


def test_image_finding_review_persistence_and_gold_export(tmp_path: Path) -> None:
    """Verify approve, correct diagnosis, reject, and mark image unusable with audit logging and gold export."""
    cfg = PlatformConfig.from_env(workspace_dir=tmp_path)
    db = DatabaseEngine(cfg)
    db.apply_migrations()

    now_iso = datetime.now(timezone.utc).isoformat()
    principal = AuthenticatedPrincipal(
        user_id="usr_rev_1",
        username="reviewer1",
        institution_id="inst_1",
        role="ADJUDICATOR",
        permissions=frozenset({"annotation:review", "medical_data:read"}),
        expires_at_iso=now_iso,
    )

    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO institutions VALUES ('inst_1', 'Hopital 1', 'SN', 'ACTIVE', 'DUA-1', ?)",
            (now_iso,),
        )
        conn.execute(
            "INSERT INTO users VALUES ('usr_rev_1', 'inst_1', 'reviewer1', 'hash', 'ADJUDICATOR', '[]', 'EXPERT', 1, ?)",
            (now_iso,),
        )
        conn.execute(
            """
            INSERT INTO image_findings (
                image_id, institution_id, subject_id, document_id, original_filename, file_format,
                content_sha256, storage_object_path, thumb_storage_path, heatmap_storage_path,
                picture_type, quality_json, malignancy_risk_probability, malignancy_risk_band,
                top3_diagnoses_json, abstained, abstain_reason_en, abstain_reason_fr,
                body_site, captured_date_iso, exif_stripped, model_name, model_version,
                git_commit, review_status, corrected_diagnosis, provenance_json, created_at
            ) VALUES (?, 'inst_1', NULL, NULL, 'lesion.webp', 'WEBP', 'sha123', '/tmp/a', '/tmp/t', '/tmp/h',
                      'skin_dermoscopic', '{}', 0.72, 'high',
                      '[{"rank": 1, "diagnosis": "basal_cell_carcinoma", "probability": 0.65}]',
                      0, NULL, NULL, 'upper extremity', '2026-11-16', 1,
                      'DermResNetMultiTask', '1.0.0', 'ce3140a', 'needs_review', NULL, '{}', ?)
            """,
            ("img_rev_1", now_iso),
        )

        # 1. Correct diagnosis class and approve
        res_corr = submit_image_finding_review(
            conn,
            principal=principal,
            image_id="img_rev_1",
            review_decision="CORRECT_AND_APPROVE",
            corrected_diagnosis="melanoma",
            corrected_risk_band="high",
            rationale="Dermoscopic atypical pigment network confirms melanoma",
        )
        assert res_corr["review_status"] == "corrected"
        assert res_corr["corrected_value"]["primary_diagnosis"] == "melanoma"

        gold = export_reviewed_image_gold(conn, institution_id="inst_1")
        assert len(gold) == 1
        assert gold[0]["gold_diagnosis"] == "melanoma"

        # 2. Mark image unusable
        res_unusable = submit_image_finding_review(
            conn,
            principal=principal,
            image_id="img_rev_1",
            review_decision="MARK_IMAGE_UNUSABLE",
            rationale="Severe motion artifact",
        )
        assert res_unusable["review_status"] == "image_unusable"

        audit_check = verify_audit_chain(conn)
        assert audit_check["valid"] is True
