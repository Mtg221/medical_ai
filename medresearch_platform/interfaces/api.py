"""FastAPI ASGI REST API, OpenAPI 3.1 schema, SQLite background jobs with SSE, and web workspace."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from medresearch_platform import __spec_coverage__, __version__
from medresearch_platform.config import PlatformConfig
from medresearch_platform.diagnostics import generate_readiness_report
from medresearch_platform.documents.ocr_engine import get_ocr_status
from medresearch_platform.documents.parsers import (
    PayloadTooLargeError,
    UnsupportedFileTypeError,
    validate_upload_payload,
)
from medresearch_platform.extraction.schema_models import (
    CLINICAL_RESEARCH_NOTICE,
    ExtractionResultSchema,
)
from medresearch_platform.interfaces.schemas import (
    BenchmarkRequest,
    IngestDocumentRequest,
    JobStatusResponse,
    ReviewEntityRequest,
    TokenRequest,
    TokenResponse,
)
from medresearch_platform.reports import ReportFormat, ReportLanguage, generate_report_bytes
from medresearch_platform.security.audit import verify_audit_chain
from medresearch_platform.security.auth import (
    AuthenticatedPrincipal,
    AuthenticationError,
    issue_access_token,
    verify_access_token,
    verify_password,
)
from medresearch_platform.workflows.jobs import (
    JobManager,
    execute_document_upload_job,
    execute_image_upload_job,
)
from medresearch_platform.workflows.pipeline import MedicalResearchPlatformService
from medresearch_platform.workflows.review import (
    export_reviewed_gold,
    export_reviewed_image_gold,
    list_human_reviews,
    submit_entity_human_review,
    submit_image_finding_review,
)


class ImageReviewRequest(BaseModel):
    review_decision: str = "APPROVE_AS_GROUND_TRUTH"
    corrected_diagnosis: str | None = None
    corrected_risk_band: str | None = None
    corrected_body_site: str | None = None
    rationale: str = "Expert dermatology review"


class ReviewSubmissionRequest(BaseModel):
    entity_id: str
    action: str  # approve | correct | reject | flag
    corrected_entity_type: str | None = None
    corrected_assertion_status: str | None = None
    corrected_temporal_status: str | None = None
    corrected_normalized_text: str | None = None
    corrected_char_start: int | None = None
    corrected_char_end: int | None = None
    corrected_surface_text: str | None = None
    rationale: str = "Expert clinical review"

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _extract_principal(request: Request, service: MedicalResearchPlatformService) -> AuthenticatedPrincipal:
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[len("Bearer ") :].strip()
        if token:
            return verify_access_token(token, service.config.token_secret_key)
    raise AuthenticationError("Missing or invalid Authorization: Bearer <token> header.")


def _auth_error_response(exc: PermissionError) -> JSONResponse:
    if isinstance(exc, AuthenticationError):
        return JSONResponse(
            {"error": str(exc)},
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )
    return JSONResponse({"error": str(exc)}, status_code=403)


def create_app(config: PlatformConfig | None = None) -> FastAPI:
    """Create the FastAPI ASGI application for the medical research platform."""
    service = MedicalResearchPlatformService(config=config)
    service.initialize_platform()
    jobs = JobManager(service.db)

    app = FastAPI(
        title="Medical Information Intelligence & Research Platform API",
        version=__version__,
        description=(
            "Research-grade bilingual (EN/FR) medical document extraction, from-scratch neural & "
            "hybrid NLP models, OCR, and longitudinal knowledge API. "
            f"{CLINICAL_RESEARCH_NOTICE}"
        ),
        openapi_url="/api/v1/openapi.json",
        docs_url="/api/v1/docs",
        debug=(service.config.profile == "development"),
    )
    app.state.service = service
    app.state.jobs = jobs

    _public_v1_paths = {"/api/v1/health", "/api/v1/auth/token", "/api/v1/openapi.json", "/api/v1/docs"}

    @app.middleware("http")
    async def require_bearer_token_middleware(request: Request, call_next: Any) -> Response:
        path = request.url.path
        if path.startswith("/api/v1/") and path not in _public_v1_paths:
            try:
                _extract_principal(request, service)
            except PermissionError as exc:
                return _auth_error_response(exc)
        return await call_next(request)

    @app.get("/api/v1/health")
    async def health_endpoint() -> JSONResponse:
        report = generate_readiness_report(service.config.workspace_dir)
        return JSONResponse({
            "status": "ok",
            "version": __version__,
            "specification_coverage": __spec_coverage__,
            "ocr": get_ocr_status(),
            "readiness": report.to_dict(),
            "clinical_notice": CLINICAL_RESEARCH_NOTICE,
        })

    @app.get("/api/v1/schema/extraction")
    async def extraction_schema_endpoint(request: Request) -> JSONResponse:
        try:
            _extract_principal(request, service)
        except PermissionError as exc:
            return _auth_error_response(exc)
        schema = ExtractionResultSchema.model_json_schema()
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        return JSONResponse(schema)

    @app.post("/api/v1/auth/token", response_model=TokenResponse)
    async def issue_token_endpoint(body: TokenRequest) -> JSONResponse:
        with service.db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ? AND is_active = 1",
                (body.username,),
            ).fetchone()
            if row is None or not verify_password(str(row["password_hash"]), body.password):
                return JSONResponse({"error": "Invalid credentials"}, status_code=401)

            perms = json.loads(str(row["permissions_json"]))
            token = issue_access_token(
                user_id=str(row["user_id"]),
                username=str(row["username"]),
                institution_id=str(row["institution_id"]),
                role=str(row["role"]),
                permissions=perms,
                secret_key=service.config.token_secret_key,
            )
            return JSONResponse({
                "access_token": token,
                "token_type": "Bearer",
                "user_id": str(row["user_id"]),
                "institution_id": str(row["institution_id"]),
                "role": str(row["role"]),
                "permissions": perms,
            })

    @app.get("/api/v1/documents")
    async def list_documents_endpoint(request: Request) -> JSONResponse:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("medical_data:read")
        except PermissionError as exc:
            return _auth_error_response(exc)

        with service.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT document_id, family_id, subject_id, institution_id, dataset_id,
                       original_filename, file_format, content_sha256, byte_size, page_count,
                       detected_language, document_type, specialty, extraction_mode,
                       untrusted_flags_json, deidentified_preview, created_at
                FROM documents
                WHERE institution_id = ?
                ORDER BY created_at DESC
                """,
                (principal.institution_id,),
            ).fetchall()
            docs = [dict(r) for r in rows]
        return JSONResponse({"documents": docs, "count": len(docs)})

    @app.get("/api/v1/documents/{document_id}")
    async def get_document_detail_endpoint(document_id: str, request: Request) -> JSONResponse:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("medical_data:read")
        except PermissionError as exc:
            return _auth_error_response(exc)

        with service.db.connect() as conn:
            doc = conn.execute("SELECT * FROM documents WHERE document_id = ?", (document_id,)).fetchone()
            if doc is None:
                return JSONResponse({"error": "Document not found"}, status_code=404)
            try:
                principal.assert_institution_access(str(doc["institution_id"]))
            except PermissionError as exc:
                return JSONResponse({"error": str(exc)}, status_code=403)

            pages = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM document_pages WHERE document_id = ? ORDER BY page_number ASC",
                    (document_id,),
                ).fetchall()
            ]
            sections = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM document_sections WHERE document_id = ? ORDER BY page_number ASC, char_start ASC",
                    (document_id,),
                ).fetchall()
            ]
            ocr_blocks = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM ocr_blocks WHERE document_id = ? ORDER BY block_index ASC",
                    (document_id,),
                ).fetchall()
            ]
            entities = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM extracted_entities WHERE document_id = ? ORDER BY char_start ASC",
                    (document_id,),
                ).fetchall()
            ]
            relations = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM extracted_relations WHERE document_id = ?",
                    (document_id,),
                ).fetchall()
            ]
            from medresearch_platform.vision.serving import _row_to_image_finding_dict

            subj_id = doc["subject_id"]
            img_rows = conn.execute(
                """
                SELECT * FROM image_findings
                WHERE institution_id = ? AND (document_id = ? OR (subject_id = ? AND subject_id IS NOT NULL))
                ORDER BY created_at DESC
                """,
                (principal.institution_id, document_id, subj_id),
            ).fetchall()
            image_findings = [_row_to_image_finding_dict(conn, r) for r in img_rows]
            same_lesion_links = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM image_text_links WHERE document_id = ? ORDER BY confidence DESC",
                    (document_id,),
                ).fetchall()
            ]

        return JSONResponse({
            "document": dict(doc),
            "pages": pages,
            "sections": sections,
            "ocr_blocks": ocr_blocks,
            "entities": entities,
            "relations": relations,
            "image_findings": image_findings,
            "same_lesion_links": same_lesion_links,
            "clinical_disclaimer": (
                "Extracted entities and relations represent AI predictions or human annotations for research. "
                "They are NOT verified clinical diagnoses or treatment orders."
            ),
        })

    @app.get("/api/v1/documents/{document_id}/report")
    async def document_report_endpoint(
        document_id: str,
        request: Request,
        format: ReportFormat = Query("markdown"),
        lang: ReportLanguage = Query("en"),
    ) -> Response:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("medical_data:read")
        except PermissionError as exc:
            return _auth_error_response(exc)

        with service.db.connect() as conn:
            doc = conn.execute("SELECT * FROM documents WHERE document_id = ?", (document_id,)).fetchone()
            if doc is None:
                return JSONResponse({"error": "Document not found"}, status_code=404)
            try:
                principal.assert_institution_access(str(doc["institution_id"]))
            except PermissionError as exc:
                return JSONResponse({"error": str(exc)}, status_code=403)

            entities = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM extracted_entities WHERE document_id = ? ORDER BY char_start ASC",
                    (document_id,),
                ).fetchall()
            ]
            relations = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM extracted_relations WHERE document_id = ?",
                    (document_id,),
                ).fetchall()
            ]
            from medresearch_platform.vision.serving import _row_to_image_finding_dict

            subj_id = doc["subject_id"]
            img_rows = conn.execute(
                """
                SELECT * FROM image_findings
                WHERE institution_id = ? AND (document_id = ? OR (subject_id = ? AND subject_id IS NOT NULL))
                ORDER BY created_at DESC
                """,
                (principal.institution_id, document_id, subj_id),
            ).fetchall()
            image_findings = [_row_to_image_finding_dict(conn, r) for r in img_rows]
            same_lesion_links = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM image_text_links WHERE document_id = ? ORDER BY confidence DESC",
                    (document_id,),
                ).fetchall()
            ]

        payload, media_type = generate_report_bytes(
            dict(doc),
            entities,
            relations,
            image_findings=image_findings,
            same_lesion_links=same_lesion_links,
            lang=lang,
            fmt=format,
        )
        return Response(content=payload, media_type=media_type)

    @app.post("/api/v1/documents/ingest")
    async def ingest_document_endpoint(body: IngestDocumentRequest, request: Request) -> JSONResponse:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("medical_data:write")
        except PermissionError as exc:
            return _auth_error_response(exc)

        if body.content_base64 is not None:
            raw_bytes = base64.b64decode(body.content_base64)
        else:
            raw_bytes = body.text.encode("utf-8")

        try:
            result = service.ingest_document(
                principal=principal,
                raw_bytes=raw_bytes,
                filename=body.filename,
                dataset_id=body.dataset_id,
                subject_raw_id=body.subject_id,
                family_id=body.family_id,
                split_assignment=body.split_assignment,
                encounter_date_iso=body.encounter_date_iso,
                specialty=body.specialty,
                document_type=body.document_type,
                force_new=body.force_new,
            )
            status_code = 200 if result.get("duplicate_of") else 201
            return JSONResponse(result, status_code=status_code)
        except PayloadTooLargeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=413)
        except UnsupportedFileTypeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=415)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.post("/api/v1/documents/upload", response_model=JobStatusResponse, status_code=202)
    async def upload_document_endpoint(
        request: Request,
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        dataset_id: str = Form("ds_synth_bilingual_v1"),
        subject_id: str = Form("subj_synth_001"),
        family_id: str = Form("fam_synth_upload"),
        split_assignment: str = Form("unassigned"),
        encounter_date_iso: str = Form("2026-06-01"),
        specialty: str = Form("INTERNAL_MEDICINE"),
        document_type: str = Form("CLINICAL_NOTE"),
        force_new: bool = Form(False),
    ) -> JSONResponse:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("medical_data:write")
        except PermissionError as exc:
            return _auth_error_response(exc)

        raw_bytes = await file.read()
        filename = file.filename or "uploaded_document.bin"
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

        job = jobs.create_job(
            institution_id=principal.institution_id,
            actor_user_id=principal.user_id,
            job_type="DOCUMENT_INGEST_AND_EXTRACT",
            original_filename=filename,
            initial_stage="QUEUED",
        )
        background_tasks.add_task(
            execute_document_upload_job,
            service,
            jobs,
            job_id=job.job_id,
            principal=principal,
            raw_bytes=raw_bytes,
            filename=filename,
            dataset_id=dataset_id,
            subject_id=subject_id,
            family_id=family_id,
            split_assignment=split_assignment,
            encounter_date_iso=encounter_date_iso,
            specialty=specialty,
            document_type=document_type,
            force_new=force_new,
        )
        return JSONResponse(job.to_dict(), status_code=202)

    @app.get("/api/v1/jobs")
    async def list_jobs_endpoint(request: Request) -> JSONResponse:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("medical_data:read")
        except PermissionError as exc:
            return _auth_error_response(exc)

        records = jobs.list_jobs(principal.institution_id)
        return JSONResponse({"jobs": [r.to_dict() for r in records], "count": len(records)})

    @app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse)
    async def get_job_endpoint(job_id: str, request: Request) -> JSONResponse:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("medical_data:read")
        except PermissionError as exc:
            return _auth_error_response(exc)

        rec = jobs.get_job(job_id)
        if rec is None:
            return JSONResponse({"error": "Job not found"}, status_code=404)
        try:
            principal.assert_institution_access(rec.institution_id)
        except PermissionError as exc:
            return _auth_error_response(exc)
        return JSONResponse(rec.to_dict())

    @app.get("/api/v1/jobs/{job_id}/events")
    async def job_events_sse_endpoint(job_id: str, request: Request) -> Any:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("medical_data:read")
        except PermissionError as exc:
            return _auth_error_response(exc)

        rec = jobs.get_job(job_id)
        if rec is None:
            return JSONResponse({"error": "Job not found"}, status_code=404)
        try:
            principal.assert_institution_access(rec.institution_id)
        except PermissionError as exc:
            return _auth_error_response(exc)

        async def event_stream() -> AsyncIterator[str]:
            last_payload: str | None = None
            for _ in range(120):
                current = jobs.get_job(job_id)
                if current is None:
                    break
                payload = json.dumps(current.to_dict())
                if payload != last_payload:
                    yield f"event: job_update\ndata: {payload}\n\n"
                    last_payload = payload
                if current.status in ("SUCCEEDED", "FAILED"):
                    break
                await asyncio.sleep(0.1)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/api/v1/entities/{entity_id}/review")
    async def review_entity_endpoint(entity_id: str, body: ReviewEntityRequest, request: Request) -> JSONResponse:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("annotation:review")
        except PermissionError as exc:
            return _auth_error_response(exc)

        try:
            with service.db.transaction() as conn:
                res = submit_entity_human_review(
                    conn,
                    principal=principal,
                    entity_id=entity_id,
                    review_decision=body.review_decision,
                    corrected_entity_type=body.corrected_entity_type,
                    corrected_assertion_status=body.corrected_assertion_status,
                    corrected_temporal_status=body.corrected_temporal_status,
                    corrected_normalized_text=body.corrected_normalized_text,
                    rationale=body.rationale,
                )
            return JSONResponse(res)
        except KeyError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        except AuthenticationError as exc:
            return _auth_error_response(exc)
        except (PermissionError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.post("/api/v1/reviews")
    async def submit_review_endpoint(body: ReviewSubmissionRequest, request: Request) -> JSONResponse:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("annotation:review")
        except PermissionError as exc:
            return _auth_error_response(exc)

        action_map = {
            "approve": "APPROVE_AS_GROUND_TRUTH",
            "correct": "CORRECT_AND_APPROVE",
            "flag": "FLAG_AMBIGUOUS",
            "reject": "REJECT",
        }
        decision = action_map.get(body.action.lower().strip(), body.action)
        try:
            with service.db.transaction() as conn:
                res = submit_entity_human_review(
                    conn,
                    principal=principal,
                    entity_id=body.entity_id,
                    review_decision=decision,
                    corrected_entity_type=body.corrected_entity_type,
                    corrected_assertion_status=body.corrected_assertion_status,
                    corrected_temporal_status=body.corrected_temporal_status,
                    corrected_normalized_text=body.corrected_normalized_text,
                    corrected_char_start=body.corrected_char_start,
                    corrected_char_end=body.corrected_char_end,
                    corrected_surface_text=body.corrected_surface_text,
                    rationale=body.rationale,
                )
            return JSONResponse(res)
        except KeyError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        except AuthenticationError as exc:
            return _auth_error_response(exc)
        except (PermissionError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.get("/api/v1/reviews")
    async def list_reviews_endpoint(
        request: Request,
        document_id: str | None = Query(default=None),
    ) -> JSONResponse:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("medical_data:read")
        except PermissionError as exc:
            return _auth_error_response(exc)

        with service.db.connect() as conn:
            reviews = list_human_reviews(
                conn,
                institution_id=principal.institution_id,
                document_id=document_id,
            )
        return JSONResponse({"reviews": reviews, "count": len(reviews)})

    @app.get("/api/v1/reviews/export")
    async def export_reviews_endpoint(
        request: Request,
        format: str = Query(default="jsonl"),
        document_id: str | None = Query(default=None),
    ) -> Response:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("medical_data:read")
        except PermissionError as exc:
            return _auth_error_response(exc)

        with service.db.connect() as conn:
            exported = export_reviewed_gold(
                conn,
                institution_id=principal.institution_id,
                fmt=format,
                document_id=document_id,
            )
        ext = "ann" if format.lower() == "brat" else "jsonl"
        return PlainTextResponse(
            exported,
            headers={"Content-Disposition": f'attachment; filename="reviewed_gold.{ext}"'},
        )

    @app.get("/api/v1/subjects/{subject_id}/timeline")
    async def subject_timeline_endpoint(subject_id: str, request: Request) -> JSONResponse:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("medical_data:read")
        except PermissionError as exc:
            return _auth_error_response(exc)

        with service.db.connect() as conn:
            subj = conn.execute("SELECT * FROM subjects WHERE subject_id = ?", (subject_id,)).fetchone()
            if subj is None:
                return JSONResponse({"error": "Subject not found"}, status_code=404)
            try:
                principal.assert_institution_access(str(subj["institution_id"]))
            except PermissionError as exc:
                return JSONResponse({"error": str(exc)}, status_code=403)

            events = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM longitudinal_events WHERE subject_id = ? ORDER BY event_date_iso ASC",
                    (subject_id,),
                ).fetchall()
            ]
        return JSONResponse({
            "subject": dict(subj),
            "timeline_events": events,
            "contradictions_count": sum(1 for e in events if int(e["contradiction_flag"]) == 1),
        })

    @app.get("/api/v1/search")
    async def search_endpoint(
        request: Request,
        q: str = Query(""),
        top_k: int = Query(5, ge=1, le=50),
    ) -> JSONResponse:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("search:query")
        except PermissionError as exc:
            return _auth_error_response(exc)

        query = q.strip()
        if not query:
            return JSONResponse({"error": "Query parameter 'q' is required."}, status_code=400)

        with service.db.connect() as conn:
            hits = service.retrieval_model.search_documents(
                conn,
                institution_id=principal.institution_id,
                query=query,
                top_k=top_k,
            )
        return JSONResponse({"query": query, "results": hits, "count": len(hits)})

    @app.get("/api/v1/terminology/map")
    async def terminology_map_endpoint(
        request: Request,
        term: str = Query(""),
        category: str | None = Query(None),
    ) -> JSONResponse:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("terminology:read")
        except PermissionError as exc:
            return _auth_error_response(exc)

        res = service.terminology.map_mention(term.strip(), category_hint=category)
        return JSONResponse({
            "source_text": res.source_text,
            "normalized_text": res.normalized_text,
            "concept_id": res.concept_id,
            "vocabulary": res.vocabulary,
            "code": res.code,
            "match_type": res.match_type,
            "confidence": res.confidence,
            "candidate_codes": list(res.candidate_codes),
            "disclaimer": "A mapped terminology code is a normalized research concept, not an approved clinical diagnosis.",
        })

    @app.get("/api/v1/models")
    async def list_models_endpoint(request: Request) -> JSONResponse:
        try:
            _extract_principal(request, service)
        except PermissionError as exc:
            return _auth_error_response(exc)

        service.ensure_models_trained_and_registered()
        with service.db.connect() as conn:
            rows = [dict(r) for r in conn.execute("SELECT * FROM models ORDER BY model_id ASC").fetchall()]
        loaded_models = service.extractor.list_loaded_models()
        return JSONResponse({"models": rows, "loaded_models": loaded_models, "count": len(rows)})

    @app.get("/api/v1/experiments")
    async def list_experiments_endpoint(request: Request) -> JSONResponse:
        try:
            _extract_principal(request, service)
        except PermissionError as exc:
            return _auth_error_response(exc)

        with service.db.connect() as conn:
            exps = [dict(r) for r in conn.execute("SELECT * FROM experiments ORDER BY created_at DESC").fetchall()]
            rounds = [dict(r) for r in conn.execute("SELECT * FROM federated_rounds ORDER BY round_number ASC").fetchall()]
        return JSONResponse({"experiments": exps, "federated_rounds": rounds})

    @app.post("/api/v1/experiments/run")
    async def run_experiments_endpoint(request: Request) -> JSONResponse:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("training:execute")
        except PermissionError as exc:
            return _auth_error_response(exc)

        manifest = service.run_phase1_experiments()
        return JSONResponse(manifest)

    @app.post("/api/v1/extraction/benchmark")
    async def run_six_extraction_benchmark_endpoint(
        request: Request,
        body: BenchmarkRequest | None = None,
    ) -> JSONResponse:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("training:execute")
        except PermissionError as exc:
            return _auth_error_response(exc)

        import json as _json
        from pathlib import Path as _Path

        epochs = body.epochs if body is not None else 15
        seed = body.seed if body is not None else 42
        v3_res = _Path("docs/results/v3_locked_test_evaluation.json")
        if v3_res.exists():
            payload = _json.loads(v3_res.read_text(encoding="utf-8"))
            return JSONResponse({"source": str(v3_res), "epochs": epochs, "seed": seed, "results": payload})
        return JSONResponse({"source": "none", "epochs": epochs, "seed": seed, "results": {}})

    @app.get("/api/v1/audit/verify")
    async def verify_audit_endpoint(request: Request) -> JSONResponse:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("audit:read")
        except PermissionError as exc:
            return _auth_error_response(exc)

        with service.db.connect() as conn:
            result = verify_audit_chain(conn)
        return JSONResponse(result)

    @app.post("/api/v1/images/upload")
    async def upload_image_endpoint(
        request: Request,
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        subject_id: str = Form("subj_synth_001"),
        document_id: str | None = Form(None),
        body_site: str | None = Form(None),
        captured_date_iso: str | None = Form(None),
        force_new: bool = Form(False),
        async_job: bool = Form(False),
    ) -> JSONResponse:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("medical_data:write")
        except PermissionError as exc:
            return _auth_error_response(exc)

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
                subject_id=subject_id,
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
            subject_id=subject_id,
            document_id=document_id,
            body_site=body_site,
            captured_date_iso=captured_date_iso,
            force_new=force_new,
        )
        completed_job = jobs.get_job(job.job_id)
        if completed_job is not None and completed_job.status == "FAILED":
            return JSONResponse({"error": completed_job.error_message, "job": completed_job.to_dict()}, status_code=400)
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

    @app.get("/api/v1/images")
    async def list_images_endpoint(request: Request) -> JSONResponse:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("medical_data:read")
        except PermissionError as exc:
            return _auth_error_response(exc)

        from medresearch_platform.vision.serving import _row_to_image_finding_dict

        with service.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM image_findings WHERE institution_id = ? ORDER BY created_at DESC",
                (principal.institution_id,),
            ).fetchall()
            items = [_row_to_image_finding_dict(conn, r) for r in rows]
        return JSONResponse({"images": items, "count": len(items)})

    @app.get("/api/v1/images/{image_id}")
    async def get_image_finding_endpoint(image_id: str, request: Request) -> JSONResponse:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("medical_data:read")
        except PermissionError as exc:
            return _auth_error_response(exc)

        from medresearch_platform.vision.serving import _row_to_image_finding_dict

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

    @app.get("/api/v1/images/{image_id}/heatmap")
    async def get_image_heatmap_endpoint(image_id: str, request: Request) -> Response:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("medical_data:read")
        except PermissionError as exc:
            return _auth_error_response(exc)

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

    @app.get("/api/v1/images/{image_id}/thumbnail")
    async def get_image_thumbnail_endpoint(image_id: str, request: Request) -> Response:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("medical_data:read")
        except PermissionError as exc:
            return _auth_error_response(exc)

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

    @app.post("/api/v1/images/{image_id}/review")
    async def review_image_finding_endpoint(
        image_id: str,
        body: ImageReviewRequest,
        request: Request,
    ) -> JSONResponse:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("annotation:review")
        except PermissionError as exc:
            return _auth_error_response(exc)

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

    @app.get("/api/v1/images/gold/export")
    async def export_image_gold_endpoint(request: Request) -> JSONResponse:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("medical_data:read")
        except PermissionError as exc:
            return _auth_error_response(exc)

        with service.db.connect() as conn:
            gold = export_reviewed_image_gold(conn, institution_id=principal.institution_id)
        return JSONResponse({"gold_images": gold, "count": len(gold)})

    @app.get("/api/v1/patients/{patient_id}/record")
    async def get_patient_record_endpoint(patient_id: str, request: Request) -> JSONResponse:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("medical_data:read")
        except PermissionError as exc:
            return _auth_error_response(exc)

        from medresearch_platform.vision.serving import get_patient_unified_record

        with service.db.connect() as conn:
            record = get_patient_unified_record(conn, principal=principal, patient_id=patient_id)
        return JSONResponse(record)

    @app.get("/api/v1/results/vision/{section}")
    async def get_vision_results_endpoint(section: str, request: Request) -> JSONResponse:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("experiment:read")
        except PermissionError as exc:
            return _auth_error_response(exc)

        from medresearch_platform.vision.serving import load_vision_benchmark_results

        data = load_vision_benchmark_results(service.config.workspace_dir)
        if section == "router":
            return JSONResponse(data["router_evaluation"])
        if section == "benchmark":
            return JSONResponse(data["skin_benchmark"])
        return JSONResponse(data)

    @app.get("/")
    async def web_workspace_endpoint() -> FileResponse:
        """Serve the accessible (WCAG 2.2 AA target) bilingual EN/FR single-page workspace."""
        return FileResponse(
            STATIC_DIR / "index.html",
            headers={
                "Content-Security-Policy": (
                    "default-src 'self'; script-src 'self'; style-src 'self'; "
                    "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
                ),
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
                "Cache-Control": "no-store",
            },
        )

    react_dist_dir = Path(__file__).resolve().parents[2] / "frontend" / "dist"

    async def react_workspace_endpoint() -> FileResponse | JSONResponse:
        """Serve the React + Vite workspace bundle if built."""
        index_file = react_dist_dir / "index.html"
        if not index_file.exists():
            return JSONResponse(
                {"error": "React frontend not built. Run: npm --prefix frontend run build"},
                status_code=404,
            )
        return FileResponse(
            index_file,
            headers={
                "Content-Security-Policy": (
                    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                    "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
                ),
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
                "Cache-Control": "no-store",
            },
        )

    app.add_api_route("/react", react_workspace_endpoint, methods=["GET"], response_model=None)
    app.add_api_route("/react/", react_workspace_endpoint, methods=["GET"], response_model=None)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    if (react_dist_dir / "assets").exists():
        app.mount("/react/assets", StaticFiles(directory=str(react_dist_dir / "assets")), name="react_assets")
    app.add_api_route("/react/{full_path:path}", react_workspace_endpoint, methods=["GET"], response_model=None)

    return app
