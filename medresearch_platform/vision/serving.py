"""Step 5: Dermatology Vision Serving Engine (EXIF/GPS stripping, bidirectional OCR/skin routing, Grad-CAM persistence, and patient record aggregation)."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from medresearch_platform.documents.parsers import (
    UnsupportedFileTypeError,
    validate_upload_payload,
)
from medresearch_platform.security.audit import record_audit_event
from medresearch_platform.security.auth import AuthenticatedPrincipal
from medresearch_platform.security.safety import inspect_untrusted_bytes
from medresearch_platform.vision.cnn_multitask import DermResNetMultiTask
from medresearch_platform.vision.linking import (
    build_image_finding,
    extract_cutaneous_text_descriptors,
    propose_same_lesion_links,
)
from medresearch_platform.vision.router import PictureTriageRouter

ALLOWED_IMAGE_FORMATS = {"PNG", "JPEG", "WEBP", "TIFF"}


def strip_exif_and_gps(raw_bytes: bytes) -> tuple[bytes, Image.Image, str | None, str]:
    """
    Parse image bytes, extract capture date (YYYY-MM-DD) if present in EXIF DateTimeOriginal (36867)
    or DateTime (306), and return `(clean_png_bytes, clean_rgb_pil_image, captured_date_iso, detected_format)`
    with 100% of EXIF, GPS, XMP, and ICC metadata stripped.
    """
    captured_date: str | None = None
    with Image.open(io.BytesIO(raw_bytes)) as im:
        fmt = (im.format or "PNG").upper()
        try:
            exif = im.getexif()
            if exif:
                for tag_id in (36867, 36868, 306):
                    raw_dt = exif.get(tag_id)
                    if isinstance(raw_dt, str):
                        m = re.search(r"(\d{4})[:\-](\d{2})[:\-](\d{2})", raw_dt)
                        if m:
                            captured_date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                            break
        except Exception:
            pass

        rgb_arr = np.asarray(im.convert("RGB"), dtype=np.uint8).copy()

    # Construct a brand-new PIL Image from raw pixel array only (zero EXIF/GPS/info dictionary)
    clean_im = Image.fromarray(rgb_arr, mode="RGB")
    clean_im.info.clear()
    buf = io.BytesIO()
    save_fmt = "WEBP" if fmt == "WEBP" else ("JPEG" if fmt == "JPEG" else "PNG")
    clean_im.save(buf, format=save_fmt)
    return buf.getvalue(), clean_im, captured_date, fmt


class ModelNotAvailableError(RuntimeError):
    """Raised when a required vision model directory, manifest, or SHA-256 checksum is missing or invalid."""

    def __init__(self, message: str, *, command: str = "medextract derm train --seed 1 --out models/derm_cnn_seed1") -> None:
        super().__init__(message)
        self.command = command


class SubjectNotFoundError(LookupError):
    """Raised when the requested subject_id does not exist in the caller's institution."""


class VisionServingEngine:
    """Lazy-loaded singleton serving `PictureTriageRouter` and `DermResNetMultiTask`."""

    def __init__(self, workspace_dir: Path | str | None = None) -> None:
        self.workspace_dir = Path(workspace_dir or Path(__file__).resolve().parents[2])
        self._router: Any = None
        self._skin_model: Any = None

    def _check_model_dir(self, model_dir: Path, numpy_weights_filename: str, torch_weights_filename: str) -> bool:
        manifest_path = model_dir / "manifest.json"
        if not manifest_path.exists():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            backend = str(manifest.get("backend") or "numpy")
            if backend == "pytorch_cpu":
                from medresearch_platform.vision.router_torch import is_torch_available

                if not is_torch_available():
                    return False
                weights_path = model_dir / torch_weights_filename
            else:
                weights_path = model_dir / numpy_weights_filename
            if not weights_path.exists():
                return False
            expected_sha = str(manifest.get("trained_weights_sha256") or "")
            if not expected_sha:
                return False
            actual_sha = hashlib.sha256(weights_path.read_bytes()).hexdigest()
            return expected_sha == actual_sha
        except Exception:
            return False

    def _router_dir(self) -> Path:
        env_dir = os.environ.get("MEDRESEARCH_VISION_ROUTER_DIR")
        return Path(env_dir) if env_dir else self.workspace_dir / "models" / "vision_router_v1"

    def _skin_dir(self) -> Path:
        env_dir = os.environ.get("MEDRESEARCH_VISION_SKIN_DIR")
        return Path(env_dir) if env_dir else self.workspace_dir / "models" / "derm_cnn_seed1"

    def status(self) -> dict[str, str]:
        r_dir = self._router_dir()
        s_dir = self._skin_dir()
        return {
            "vision_router": "ready" if self._check_model_dir(r_dir, "router_weights.npz", "router_torch.pt") else "missing",
            "vision_skin": "ready" if self._check_model_dir(s_dir, "weights.npz", "weights_torch.pt") else "missing",
        }

    @property
    def router(self) -> Any:
        if self._router is None:
            r_dir = self._router_dir()
            try:
                manifest = json.loads((r_dir / "manifest.json").read_text(encoding="utf-8"))
                if manifest.get("backend") == "pytorch_cpu":
                    from medresearch_platform.vision.router_torch import (
                        PyTorchPictureTriageRouter,
                        is_torch_available,
                    )

                    if not is_torch_available():
                        raise RuntimeError("PyTorch is not installed for pytorch_cpu router model.")
                    self._router = PyTorchPictureTriageRouter.load(r_dir)
                else:
                    self._router = PictureTriageRouter.load(r_dir)
            except Exception as exc:
                raise ModelNotAvailableError(
                    f"image model not trained (vision_router_v1 missing or invalid in {r_dir}): {exc} / "
                    "Modèle image non entraîné (vision_router_v1 manquant ou invalide).",
                    command="medextract derm train --seed 1 --out models/derm_cnn_seed1",
                ) from exc
        return self._router

    @property
    def skin_model(self) -> Any:
        if self._skin_model is None:
            m_dir = self._skin_dir()
            try:
                manifest = json.loads((m_dir / "manifest.json").read_text(encoding="utf-8"))
                if manifest.get("backend") == "pytorch_cpu":
                    from medresearch_platform.vision.cnn_torch import PyTorchDermResNet
                    from medresearch_platform.vision.router_torch import is_torch_available

                    if not is_torch_available():
                        raise RuntimeError("PyTorch is not installed for pytorch_cpu skin model.")
                    self._skin_model = PyTorchDermResNet.load(m_dir)
                else:
                    self._skin_model = DermResNetMultiTask.load(m_dir)
            except Exception as exc:
                raise ModelNotAvailableError(
                    f"image model not trained (derm_cnn_seed1 missing or invalid in {m_dir}): {exc} / "
                    "Modèle image non entraîné (derm_cnn_seed1 manquant ou invalide).",
                    command="medextract derm train --seed 1 --out models/derm_cnn_seed1",
                ) from exc
        return self._skin_model


def ingest_and_analyze_image(
    service: Any,
    vision_engine: VisionServingEngine,
    *,
    principal: AuthenticatedPrincipal,
    raw_bytes: bytes,
    filename: str,
    subject_id: str | None = None,
    document_id: str | None = None,
    body_site: str | None = None,
    captured_date_iso: str | None = None,
    force_new: bool = False,
    allow_document_reroute: bool = True,
    progress_callback: Callable[[str, int], None] | None = None,
) -> dict[str, Any]:
    """
    Full Step 5 image upload pipeline:
    1. Validate magic bytes, size limit, subject existence, and vision model availability; strip 100% of EXIF & GPS metadata before storage.
    2. Stage `TRIAGE`: run `PictureTriageRouter`. If `document_scan` and `allow_document_reroute=True`,
       automatically route to `service.ingest_document` (OCR + text pipeline).
    3. Stage `QUALITY`: run bilingual quality check.
    4. Stage `ANALYSE`: run `DermResNetMultiTask` for calibrated risk, top-3 diagnoses, and abstention.
    5. Stage `EXPLAIN`: compute Grad-CAM overlay PNG, persist `ImageFinding` and `SAME_LESION` text links.
    """
    principal.assert_permission("medical_data:write")
    if not subject_id or not str(subject_id).strip():
        raise ValueError("subject_id is required / subject_id est requis")
    subject_id = str(subject_id).strip()

    validate_upload_payload(
        raw_bytes,
        filename,
        max_upload_bytes=service.config.max_upload_bytes,
    )
    safety = inspect_untrusted_bytes(
        raw_bytes,
        filename,
        max_upload_bytes=service.config.max_upload_bytes,
    )
    if not safety.is_safe_to_parse:
        raise ValueError(f"Image rejected by safety guard: {safety.rejection_reason}")
    if safety.detected_format not in ALLOWED_IMAGE_FORMATS:
        raise UnsupportedFileTypeError(
            f"Image endpoint requires PNG, JPEG, WEBP, or TIFF (got {safety.detected_format})."
        )

    # Verify subject exists in caller's institution (V9: never fall back to another patient)
    with service.db.connect() as conn:
        s_row = conn.execute(
            "SELECT subject_id FROM subjects WHERE institution_id = ? AND (subject_id = ? OR pseudonym_code = ?) LIMIT 1",
            (principal.institution_id, subject_id, subject_id),
        ).fetchone()
        if s_row is None:
            raise SubjectNotFoundError(
                f"Unknown subject_id '{subject_id}' for institution '{principal.institution_id}'."
            )
        resolved_subj_id = str(s_row["subject_id"])

    # Fail loudly (503) if vision models are missing or manifest SHA-256 fails (V8)
    router_model = vision_engine.router

    clean_bytes, clean_im, exif_date, _ = strip_exif_and_gps(raw_bytes)
    user_date = captured_date_iso.strip() if isinstance(captured_date_iso, str) and captured_date_iso.strip() else None
    if user_date is not None:
        resolved_date: str | None = user_date
        captured_date_source = "user"
    elif exif_date is not None:
        resolved_date = exif_date
        captured_date_source = "exif"
    else:
        resolved_date = None
        captured_date_source = "unknown"

    warnings: list[str] = []
    if captured_date_source == "unknown":
        warnings.append(
            "Date missing: no EXIF or user capture date provided / Date manquante : aucune date EXIF ou saisie fournie"
        )

    content_sha256 = hashlib.sha256(clean_bytes).hexdigest()

    # Duplicate check within caller's institution
    if not force_new:
        with service.db.connect() as conn:
            existing = conn.execute(
                "SELECT * FROM image_findings WHERE institution_id = ? AND content_sha256 = ? ORDER BY created_at ASC LIMIT 1",
                (principal.institution_id, content_sha256),
            ).fetchone()
            if existing is not None:
                ex_id = str(existing["image_id"])
                return _row_to_image_finding_dict(conn, existing, is_duplicate=True, duplicate_of=ex_id)

    if progress_callback:
        progress_callback("TRIAGE", 20)

    triage = router_model.triage_image(clean_im)

    # Bidirectional routing: if a document photo is uploaded to /api/v1/images/upload, route to OCR + text pipeline
    if triage.predicted_class == "document_scan" and allow_document_reroute:
        doc_res = service.ingest_document(
            principal=principal,
            raw_bytes=clean_bytes,
            filename=filename,
            dataset_id="ds_synth_bilingual_v1",
            subject_raw_id=resolved_subj_id,
            family_id="fam_synth_upload",
            encounter_date_iso=resolved_date,
            force_new=force_new,
            _skip_skin_reroute=True,
        )
        return {
            "routed_to": "ocr_text_pipeline",
            "triage": triage.to_dict(),
            "document_result": doc_res,
            "exif_stripped": True,
        }

    # Check skin model availability before writing any files to storage (V8)
    skin_model = vision_engine.skin_model

    if progress_callback:
        progress_callback("QUALITY", 45)

    skin_pred = None
    if triage.route_action == "skin_vision_pipeline":
        if progress_callback:
            progress_callback("ANALYSE", 75)
        skin_pred = skin_model.predict_image(clean_im)

    if progress_callback:
        progress_callback("EXPLAIN", 90)

    image_id = f"img_{uuid.uuid4().hex[:14]}"
    orig_dir = service.config.storage_dir / "originals"
    derived_dir = service.config.storage_dir / "derived"
    orig_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)

    ext = safety.detected_format.lower()
    orig_path = orig_dir / f"{image_id}_{content_sha256[:16]}.{ext}"
    orig_path.write_bytes(clean_bytes)

    # Save thumbnail (160x160 WEBP, stripped of EXIF)
    thumb_path = derived_dir / f"{image_id}_thumb.webp"
    thumb_im = clean_im.copy()
    thumb_im.thumbnail((160, 160), Image.Resampling.BILINEAR)
    thumb_im.save(thumb_path, format="WEBP")

    # Save Grad-CAM heatmap overlay PNG
    heatmap_path = derived_dir / f"{image_id}_heatmap.png"
    if skin_pred is not None:
        cam_grid = np.asarray(skin_pred.heatmap_grid, dtype=np.float64)
    else:
        cam_grid, _ = skin_model.compute_gradcam_heatmap(clean_im)
    heatmap_bytes = skin_model.render_gradcam_overlay_png(clean_im, cam_grid)
    heatmap_path.write_bytes(heatmap_bytes)

    finding = build_image_finding(
        image_id=image_id,
        document_id=document_id,
        patient_id=resolved_subj_id,
        triage=triage,
        skin_prediction=skin_pred,
        body_site=body_site,
        captured_date_iso=resolved_date,
        heatmap_uri=f"/api/v1/images/{image_id}/heatmap",
    )
    finding.provenance["captured_date_source"] = captured_date_source
    finding.provenance["warnings"] = warnings

    now_iso = datetime.now(timezone.utc).isoformat()
    created_links: list[dict[str, Any]] = []

    with service.db.transaction() as conn:

        conn.execute(
            """
            INSERT INTO image_findings (
                image_id, institution_id, subject_id, document_id, original_filename, file_format,
                content_sha256, storage_object_path, thumb_storage_path, heatmap_storage_path,
                picture_type, quality_json, malignancy_risk_probability, malignancy_risk_band,
                top3_diagnoses_json, abstained, abstain_reason_en, abstain_reason_fr,
                body_site, captured_date_iso, exif_stripped, model_name, model_version,
                git_commit, review_status, corrected_diagnosis, provenance_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, NULL, ?, ?)
            """,
            (
                finding.image_id,
                principal.institution_id,
                resolved_subj_id,
                document_id,
                filename,
                safety.detected_format,
                content_sha256,
                str(orig_path),
                str(thumb_path),
                str(heatmap_path),
                finding.picture_type,
                json.dumps(finding.quality_result),
                finding.malignancy_risk_probability,
                finding.malignancy_risk_band,
                json.dumps([dict(d) for d in finding.top3_diagnoses]),
                1 if finding.abstained else 0,
                finding.abstain_reason_en,
                finding.abstain_reason_fr,
                finding.body_site,
                finding.captured_date_iso,
                finding.model_name,
                finding.model_version,
                finding.git_commit,
                finding.review_status,
                json.dumps(finding.provenance),
                now_iso,
            ),
        )

        # Link to text entities across patient's documents (or target document_id)
        doc_query = "SELECT * FROM documents WHERE institution_id = ?"
        doc_params: list[Any] = [principal.institution_id]
        if document_id:
            doc_query += " AND document_id = ?"
            doc_params.append(document_id)
        elif resolved_subj_id:
            doc_query += " AND subject_id = ?"
            doc_params.append(resolved_subj_id)

        candidate_docs = conn.execute(doc_query, tuple(doc_params)).fetchall()
        for doc_row in candidate_docs:
            d_id = str(doc_row["document_id"])
            pages = conn.execute(
                "SELECT raw_text FROM document_pages WHERE document_id = ? ORDER BY page_number ASC",
                (d_id,),
            ).fetchall()
            full_text = "\n".join(str(p["raw_text"]) for p in pages) or str(doc_row["deidentified_preview"] or "")
            existing_ents = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM extracted_entities WHERE document_id = ? ORDER BY char_start ASC",
                    (d_id,),
                ).fetchall()
            ]
            # Persist any newly extracted cutaneous anatomy/lesion descriptors into extracted_entities
            new_cut_ents = extract_cutaneous_text_descriptors(full_text, existing_entities=existing_ents)
            for ce in new_cut_ents:
                new_eid = f"ent_{uuid.uuid4().hex[:14]}"
                ce["entity_id"] = new_eid
                conn.execute(
                    """
                    INSERT INTO extracted_entities (
                        entity_id, document_id, page_number, section_type, entity_type,
                        source_text, normalized_text, terminology_concept_id, mapping_match_type,
                        assertion_status, temporal_status, medication_status, char_start, char_end,
                        bbox_json, model_id, model_version, source_stage, confidence, uncertainty,
                        abstained, review_status, review_reason, created_at
                    ) VALUES (?, ?, 1, 'BODY', ?, ?, ?, NULL, 'UNMAPPED', 'AFFIRMED', 'CURRENT',
                              'NOT_APPLICABLE', ?, ?, '{}', 'cutaneous_descriptor_v1', '1.0.0',
                              'cutaneous_descriptor_extractor', ?, 0.08, 0, 'AI_PREDICTED', '', ?)
                    """,
                    (
                        new_eid,
                        d_id,
                        ce["entity_type"],
                        ce["source_text"],
                        ce["normalized_text"],
                        ce["char_start"],
                        ce["char_end"],
                        ce["confidence"],
                        now_iso,
                    ),
                )
                existing_ents.append(ce)

            proposed = propose_same_lesion_links(
                full_text,
                existing_ents,
                [finding],
                document_id=d_id,
                patient_id=resolved_subj_id,
                document_date_iso=resolved_date,
            )
            for lnk in proposed:
                conn.execute(
                    """
                    INSERT INTO image_text_links (
                        link_id, institution_id, subject_id, document_id, image_id,
                        text_entity_id, text_span, char_start, char_end, matched_body_site,
                        date_delta_days, confidence, review_status, rationale_en, rationale_fr, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        lnk["link_id"],
                        principal.institution_id,
                        resolved_subj_id,
                        d_id,
                        finding.image_id,
                        lnk["text_entity_id"],
                        lnk["text_span"],
                        lnk["char_start"],
                        lnk["char_end"],
                        lnk["matched_body_site"],
                        lnk["date_delta_days"],
                        lnk["confidence"],
                        lnk["review_status"],
                        lnk["rationale_en"],
                        lnk["rationale_fr"],
                        now_iso,
                    ),
                )
                created_links.append(lnk)

        record_audit_event(
            conn,
            actor_user_id=principal.user_id,
            institution_id=principal.institution_id,
            action_type="IMAGE_INGEST_AND_ANALYZE",
            resource_type="IMAGE_FINDING",
            resource_id=finding.image_id,
            metadata={
                "picture_type": finding.picture_type,
                "risk_band": finding.malignancy_risk_band,
                "abstained": finding.abstained,
                "exif_stripped": True,
                "same_lesion_links_count": len(created_links),
            },
        )

    out = finding.to_dict()
    out["routed_to"] = (
        "skin_vision_pipeline" if triage.route_action == "skin_vision_pipeline" else "reject_unsupported"
    )
    out["triage"] = triage.to_dict()
    out["exif_stripped"] = True
    out["is_duplicate"] = False
    out["captured_date_source"] = captured_date_source
    out["warnings"] = warnings
    out["same_lesion_links"] = created_links
    out["thumbnail_uri"] = f"/api/v1/images/{image_id}/thumbnail"
    return out


def _row_to_image_finding_dict(
    conn: Any,
    row: Any,
    *,
    is_duplicate: bool = False,
    duplicate_of: str | None = None,
) -> dict[str, Any]:
    img_id = str(row["image_id"])
    links = [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM image_text_links WHERE image_id = ? ORDER BY confidence DESC",
            (img_id,),
        ).fetchall()
    ]
    top3 = json.loads(str(row["top3_diagnoses_json"] or "[]"))
    quality = json.loads(str(row["quality_json"] or "{}"))
    prov = json.loads(str(row["provenance_json"] or "{}"))
    return {
        "image_id": img_id,
        "document_id": row["document_id"],
        "patient_id": row["subject_id"],
        "original_filename": str(row["original_filename"]),
        "file_format": str(row["file_format"]),
        "picture_type": str(row["picture_type"]),
        "quality_result": quality,
        "malignancy_risk_probability": row["malignancy_risk_probability"],
        "malignancy_risk_band": str(row["malignancy_risk_band"]),
        "top3_diagnoses": top3,
        "corrected_diagnosis": row["corrected_diagnosis"],
        "heatmap_uri": f"/api/v1/images/{img_id}/heatmap",
        "thumbnail_uri": f"/api/v1/images/{img_id}/thumbnail",
        "abstained": bool(row["abstained"]),
        "abstain_reason_en": row["abstain_reason_en"],
        "abstain_reason_fr": row["abstain_reason_fr"],
        "body_site": row["body_site"],
        "captured_date_iso": row["captured_date_iso"],
        "captured_date_source": str(prov.get("captured_date_source") or ("exif" if row["captured_date_iso"] else "unknown")),
        "warnings": list(prov.get("warnings") or ([] if row["captured_date_iso"] else [
            "Date missing: no EXIF or user capture date provided / Date manquante : aucune date EXIF ou saisie fournie"
        ])),
        "exif_stripped": bool(row["exif_stripped"]),
        "model_name": str(row["model_name"]),
        "model_version": str(row["model_version"]),
        "git_commit": str(row["git_commit"]),
        "review_status": str(row["review_status"]),
        "provenance": prov,
        "safety_notice_en": "Research tool — not a diagnosis. A dermatologist must examine the lesion.",
        "safety_notice_fr": "Outil de recherche — ne constitue pas un diagnostic. Un dermatologue doit examiner la lésion.",
        "same_lesion_links": links,
        "is_duplicate": is_duplicate,
        "duplicate_of": duplicate_of,
        "created_at": str(row["created_at"]),
    }


def get_patient_unified_record(
    conn: Any,
    *,
    principal: AuthenticatedPrincipal,
    patient_id: str,
) -> dict[str, Any]:
    """
    Return the unified patient record (`GET /api/v1/patients/{id}/record`):
    text documents, extracted entities, image findings, `SAME_LESION` cross-modal links,
    and chronological timeline events (text facts + skin images).
    """
    principal.assert_permission("medical_data:read")
    subj = conn.execute(
        "SELECT * FROM subjects WHERE institution_id = ? AND (subject_id = ? OR pseudonym_code = ?) LIMIT 1",
        (principal.institution_id, patient_id, patient_id),
    ).fetchone()

    resolved_subj_id = str(subj["subject_id"]) if subj is not None else patient_id

    docs = [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM documents WHERE institution_id = ? AND subject_id = ? ORDER BY created_at DESC",
            (principal.institution_id, resolved_subj_id),
        ).fetchall()
    ]
    doc_ids = [str(d["document_id"]) for d in docs]

    entities: list[dict[str, Any]] = []
    for d_id in doc_ids:
        entities.extend(
            dict(r)
            for r in conn.execute(
                "SELECT * FROM extracted_entities WHERE document_id = ? ORDER BY char_start ASC",
                (d_id,),
            ).fetchall()
        )

    img_rows = conn.execute(
        "SELECT * FROM image_findings WHERE institution_id = ? AND subject_id = ? ORDER BY created_at DESC",
        (principal.institution_id, resolved_subj_id),
    ).fetchall()
    image_findings = [_row_to_image_finding_dict(conn, r) for r in img_rows]

    links = [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM image_text_links WHERE institution_id = ? AND subject_id = ? ORDER BY confidence DESC",
            (principal.institution_id, resolved_subj_id),
        ).fetchall()
    ]

    timeline = [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM longitudinal_events WHERE subject_id = ? ORDER BY event_date_iso ASC",
            (resolved_subj_id,),
        ).fetchall()
    ]
    for f in image_findings:
        top_dx = (
            f["corrected_diagnosis"]
            or (f["top3_diagnoses"][0]["diagnosis"] if f["top3_diagnoses"] else "abstained")
        )
        cap_date = f.get("captured_date_iso")
        timeline.append({
            "event_id": f"evt_{f['image_id']}",
            "subject_id": resolved_subj_id,
            "document_id": f["document_id"],
            "image_id": f["image_id"],
            "event_category": "SKIN_IMAGE_FINDING",
            "summary_text": (
                f"[{f['picture_type']}] {top_dx} — risk band: {f['malignancy_risk_band']} "
                f"(site: {f['body_site'] or 'unspecified'})"
            ),
            "assertion_status": "ABSTAINED" if f["abstained"] else "AFFIRMED",
            "temporal_status": "CURRENT",
            "event_date_iso": str(cap_date) if cap_date else None,
            "date_unknown": cap_date is None,
            "derivation_type": "EXTRACTED",
            "confidence": float(f["malignancy_risk_probability"] or 0.5),
            "review_status": f["review_status"],
        })
    timeline.sort(key=lambda e: (e.get("event_date_iso") is None, str(e.get("event_date_iso") or "")))

    return {
        "patient_id": resolved_subj_id,
        "pseudonym_code": str(subj["pseudonym_code"]) if subj is not None else patient_id,
        "institution_id": principal.institution_id,
        "documents": docs,
        "text_entities": entities,
        "image_findings": image_findings,
        "same_lesion_links": links,
        "timeline_events": timeline,
        "safety_notice_en": "Research tool — not a diagnosis. A dermatologist must examine the lesion.",
        "safety_notice_fr": "Outil de recherche — ne constitue pas un diagnostic. Un dermatologue doit examiner la lésion.",
    }


def load_vision_benchmark_results(workspace_dir: Path | str | None = None) -> dict[str, Any]:
    """Load real vision evaluation JSON files from `docs/results/vision/*.json`."""
    base = Path(workspace_dir or Path(__file__).resolve().parents[2])
    router_path = base / "docs" / "results" / "vision" / "step2_router_eval.json"
    skin_path = base / "docs" / "results" / "vision" / "step3_skin_benchmark.json"

    router_data = json.loads(router_path.read_text(encoding="utf-8")) if router_path.exists() else {}
    skin_data = json.loads(skin_path.read_text(encoding="utf-8")) if skin_path.exists() else {}

    return {
        "router_evaluation": router_data,
        "skin_benchmark": skin_data,
        "pytorch_available": bool(
            __import__("medresearch_platform.vision.cnn_multitask", fromlist=["is_pytorch_available"]).is_pytorch_available()
        ),
        "safety_notice_en": "Research tool — not a diagnosis. A dermatologist must examine the lesion.",
        "safety_notice_fr": "Outil de recherche — ne constitue pas un diagnostic. Un dermatologue doit examiner la lésion.",
    }
