"""Dermatology vision upload, triage, review, patient record, and vision benchmark routes."""

from __future__ import annotations

import base64
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from medresearch_platform.documents.parsers import (
    PayloadTooLargeError,
    UnsupportedFileTypeError,
    validate_upload_payload,
)
from medresearch_platform.interfaces.routers.common import (
    ImageReviewRequest,
    auth_error_response,
    extract_principal,
)
from medresearch_platform.vision.serving import (
    ModelNotAvailableError,
    _row_to_image_finding_dict,
    get_patient_unified_record,
    load_vision_benchmark_results,
)
from medresearch_platform.workflows.jobs import JobManager, execute_image_upload_job
from medresearch_platform.workflows.pipeline import MedicalResearchPlatformService
from medresearch_platform.workflows.review import (
    export_reviewed_image_gold,
    submit_image_finding_review,
)


def build_vision_router(
    service: MedicalResearchPlatformService,
    jobs: JobManager,
) -> APIRouter:
    router = APIRouter(tags=["vision"])

    @router.post("/api/v1/images/upload")
    async def upload_image_endpoint(
        request: Request,
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        subject_id: str = Form(...),
        document_id: str | None = Form(None),
        body_site: str | None = Form(None),
        captured_date_iso: str | None = Form(None),
        force_new: bool = Form(False),
        async_job: bool = Form(False),
    ) -> JSONResponse:
        try:
            principal = extract_principal(request, service)
            principal.assert_permission("medical_data:write")
        except PermissionError as exc:
            return auth_error_response(exc)

        raw_bytes = await file.read()
        filename = file.filename or "uploaded_image.png"
        try:
            validate_upload_payload(
                raw_bytes,
                filename,
                max_upload_bytes=service.config.max_upload_bytes,
            )
        except PayloadTooLargeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=413)
        except UnsupportedFileTypeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=415)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        try:
            _ = service.vision_engine.router
            _ = service.vision_engine.skin_model
        except ModelNotAvailableError as exc:
            return JSONResponse(
                {
                    "error": "image model not trained / modèle image non entraîné",
                    "detail": str(exc),
                    "command": exc.command,
                },
                status_code=503,
            )

        subj_clean = subject_id.strip()
        if not subj_clean:
            return JSONResponse({"error": "subject_id is required / subject_id est requis"}, status_code=422)
        with service.db.connect() as conn:
            s_row = conn.execute(
                "SELECT subject_id FROM subjects WHERE institution_id = ? AND (subject_id = ? OR pseudonym_code = ?) LIMIT 1",
                (principal.institution_id, subj_clean, subj_clean),
            ).fetchone()
            if s_row is None:
                return JSONResponse(
                    {"error": f"Unknown subject_id '{subj_clean}' for institution '{principal.institution_id}'."},
                    status_code=404,
                )

        job = jobs.create_job(
            institution_id=principal.institution_id,
            actor_user_id=principal.user_id,
            job_type="IMAGE_TRIAGE_AND_ANALYZE",
            original_filename=filename,
            initial_stage="TRIAGE",
        )
        if async_job:
            background_tasks.add_task(
                execute_image_upload_job,
                service,
                jobs,
                job_id=job.job_id,
                principal=principal,
                raw_bytes=raw_bytes,
                filename=filename,
                subject_id=subj_clean,
                document_id=document_id,
                body_site=body_site,
                captured_date_iso=captured_date_iso,
                force_new=force_new,
            )
            return JSONResponse(job.to_dict(), status_code=202)

        execute_image_upload_job(
            service,
            jobs,
            job_id=job.job_id,
            principal=principal,
            raw_bytes=raw_bytes,
            filename=filename,
            subject_id=subj_clean,
            document_id=document_id,
            body_site=body_site,
            captured_date_iso=captured_date_iso,
            force_new=force_new,
        )
        completed_job = jobs.get_job(job.job_id)
        if completed_job is not None and completed_job.status == "FAILED":
            err_msg = str(completed_job.error_message or "")
            if "image model not trained" in err_msg.lower():
                return JSONResponse(
                    {
                        "error": "image model not trained / modèle image non entraîné",
                        "detail": err_msg,
                        "command": "medextract derm train --seed 1 --out models/derm_cnn_seed1",
                        "job": completed_job.to_dict(),
                    },
                    status_code=503,
                )
            if "unknown subject_id" in err_msg.lower():
                return JSONResponse({"error": err_msg, "job": completed_job.to_dict()}, status_code=404)
            return JSONResponse({"error": err_msg, "job": completed_job.to_dict()}, status_code=400)
        res_payload = completed_job.result if completed_job and completed_job.result else {}
        return JSONResponse(
            {
                "job_id": job.job_id,
                "stages": ["TRIAGE", "QUALITY", "ANALYSE", "EXPLAIN"],
                "status": "SUCCEEDED",
                **res_payload,
            },
            status_code=201,
        )

    @router.get("/api/v1/images")
    async def list_images_endpoint(request: Request) -> JSONResponse:
        try:
            principal = extract_principal(request, service)
            principal.assert_permission("medical_data:read")
        except PermissionError as exc:
            return auth_error_response(exc)

        with service.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM image_findings WHERE institution_id = ? ORDER BY created_at DESC",
                (principal.institution_id,),
            ).fetchall()
            items = [_row_to_image_finding_dict(conn, r) for r in rows]
        return JSONResponse({"images": items, "count": len(items)})

    @router.get("/api/v1/images/gold/export")
    async def export_image_gold_endpoint(request: Request) -> JSONResponse:
        try:
            principal = extract_principal(request, service)
            principal.assert_permission("medical_data:read")
        except PermissionError as exc:
            return auth_error_response(exc)

        with service.db.connect() as conn:
            gold = export_reviewed_image_gold(conn, institution_id=principal.institution_id)
        return JSONResponse({"gold_images": gold, "count": len(gold)})

    @router.get("/api/v1/images/{image_id}")
    async def get_image_finding_endpoint(image_id: str, request: Request) -> JSONResponse:
        try:
            principal = extract_principal(request, service)
            principal.assert_permission("medical_data:read")
        except PermissionError as exc:
            return auth_error_response(exc)

        with service.db.connect() as conn:
            row = conn.execute("SELECT * FROM image_findings WHERE image_id = ?", (image_id,)).fetchone()
            if row is None:
                return JSONResponse({"error": "Image finding not found"}, status_code=404)
            try:
                principal.assert_institution_access(str(row["institution_id"]))
            except PermissionError as exc:
                return JSONResponse({"error": str(exc)}, status_code=403)
            payload = _row_to_image_finding_dict(conn, row)
            thumb_p = Path(str(row["thumb_storage_path"] or ""))
            if thumb_p.exists():
                payload["thumbnail_base64"] = base64.b64encode(thumb_p.read_bytes()).decode("ascii")
        return JSONResponse(payload)

    @router.get("/api/v1/images/{image_id}/heatmap")
    async def get_image_heatmap_endpoint(image_id: str, request: Request) -> Response:
        try:
            principal = extract_principal(request, service)
            principal.assert_permission("medical_data:read")
        except PermissionError as exc:
            return auth_error_response(exc)

        with service.db.connect() as conn:
            row = conn.execute("SELECT * FROM image_findings WHERE image_id = ?", (image_id,)).fetchone()
            if row is None:
                return JSONResponse({"error": "Image finding not found"}, status_code=404)
            try:
                principal.assert_institution_access(str(row["institution_id"]))
            except PermissionError as exc:
                return JSONResponse({"error": str(exc)}, status_code=403)
            heatmap_p = Path(str(row["heatmap_storage_path"] or ""))
            if not heatmap_p.exists():
                return JSONResponse({"error": "Heatmap file not found"}, status_code=404)
            return Response(content=heatmap_p.read_bytes(), media_type="image/png")

    @router.get("/api/v1/images/{image_id}/thumbnail")
    async def get_image_thumbnail_endpoint(image_id: str, request: Request) -> Response:
        try:
            principal = extract_principal(request, service)
            principal.assert_permission("medical_data:read")
        except PermissionError as exc:
            return auth_error_response(exc)

        with service.db.connect() as conn:
            row = conn.execute("SELECT * FROM image_findings WHERE image_id = ?", (image_id,)).fetchone()
            if row is None:
                return JSONResponse({"error": "Image finding not found"}, status_code=404)
            try:
                principal.assert_institution_access(str(row["institution_id"]))
            except PermissionError as exc:
                return JSONResponse({"error": str(exc)}, status_code=403)
            thumb_p = Path(str(row["thumb_storage_path"] or ""))
            if not thumb_p.exists():
                return JSONResponse({"error": "Thumbnail file not found"}, status_code=404)
            return Response(content=thumb_p.read_bytes(), media_type="image/webp")

    @router.post("/api/v1/images/{image_id}/review")
    async def review_image_finding_endpoint(
        image_id: str,
        body: ImageReviewRequest,
        request: Request,
    ) -> JSONResponse:
        try:
            principal = extract_principal(request, service)
            principal.assert_permission("annotation:review")
        except PermissionError as exc:
            return auth_error_response(exc)

        try:
            with service.db.transaction() as conn:
                result = submit_image_finding_review(
                    conn,
                    principal=principal,
                    image_id=image_id,
                    review_decision=body.review_decision,
                    corrected_diagnosis=body.corrected_diagnosis,
                    corrected_risk_band=body.corrected_risk_band,
                    corrected_body_site=body.corrected_body_site,
                    rationale=body.rationale,
                )
            return JSONResponse(result)
        except KeyError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        except PermissionError as exc:
            return JSONResponse({"error": str(exc)}, status_code=403)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @router.get("/api/v1/patients/{patient_id}/record")
    async def get_patient_record_endpoint(patient_id: str, request: Request) -> JSONResponse:
        try:
            principal = extract_principal(request, service)
            principal.assert_permission("medical_data:read")
        except PermissionError as exc:
            return auth_error_response(exc)

        with service.db.connect() as conn:
            record = get_patient_unified_record(conn, principal=principal, patient_id=patient_id)
        return JSONResponse(record)

    @router.get("/api/v1/results/vision/{section}")
    async def get_vision_results_endpoint(section: str, request: Request) -> JSONResponse:
        try:
            principal = extract_principal(request, service)
            principal.assert_permission("experiment:read")
        except PermissionError as exc:
            return auth_error_response(exc)

        data = load_vision_benchmark_results(service.config.workspace_dir)
        if section == "router":
            return JSONResponse(data["router_evaluation"])
        if section == "benchmark":
            return JSONResponse(data["skin_benchmark"])
        return JSONResponse(data)

    return router
