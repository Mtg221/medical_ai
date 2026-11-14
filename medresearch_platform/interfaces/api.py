"""FastAPI ASGI REST API, OpenAPI 3.1 schema, SQLite background jobs with SSE, and web workspace."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from medresearch_platform import __spec_coverage__, __version__
from medresearch_platform.config import PlatformConfig
from medresearch_platform.diagnostics import generate_readiness_report
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
from medresearch_platform.security.audit import verify_audit_chain
from medresearch_platform.security.auth import (
    AuthenticatedPrincipal,
    issue_access_token,
    verify_access_token,
    verify_password,
)
from medresearch_platform.workflows.jobs import JobManager, execute_document_upload_job
from medresearch_platform.workflows.pipeline import MedicalResearchPlatformService
from medresearch_platform.workflows.review import submit_entity_human_review

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _extract_principal(request: Request, service: MedicalResearchPlatformService) -> AuthenticatedPrincipal:
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[len("Bearer ") :].strip()
        return verify_access_token(token, service.config.token_secret_key)
    raise PermissionError("Missing or invalid Authorization: Bearer <token> header.")


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

    @app.get("/api/v1/health")
    async def health_endpoint() -> JSONResponse:
        report = generate_readiness_report(service.config.workspace_dir)
        return JSONResponse({
            "status": "ok",
            "version": __version__,
            "specification_coverage": __spec_coverage__,
            "readiness": report.to_dict(),
            "clinical_notice": CLINICAL_RESEARCH_NOTICE,
        })

    @app.get("/api/v1/schema/extraction")
    async def extraction_schema_endpoint() -> JSONResponse:
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
            return JSONResponse({"error": str(exc)}, status_code=403)

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
            return JSONResponse({"error": str(exc)}, status_code=403)

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

        return JSONResponse({
            "document": dict(doc),
            "pages": pages,
            "sections": sections,
            "ocr_blocks": ocr_blocks,
            "entities": entities,
            "relations": relations,
            "clinical_disclaimer": (
                "Extracted entities and relations represent AI predictions or human annotations for research. "
                "They are NOT verified clinical diagnoses or treatment orders."
            ),
        })

    @app.post("/api/v1/documents/ingest")
    async def ingest_document_endpoint(body: IngestDocumentRequest, request: Request) -> JSONResponse:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("medical_data:write")
        except PermissionError as exc:
            return JSONResponse({"error": str(exc)}, status_code=403)

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
            )
            return JSONResponse(result, status_code=201)
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
    ) -> JSONResponse:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("medical_data:write")
        except PermissionError as exc:
            return JSONResponse({"error": str(exc)}, status_code=403)

        raw_bytes = await file.read()
        filename = file.filename or "uploaded_document.bin"
        job = jobs.create_job(
            institution_id=principal.institution_id,
            actor_user_id=principal.user_id,
            job_type="DOCUMENT_INGEST_AND_EXTRACT",
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
        )
        return JSONResponse(job.to_dict(), status_code=202)

    @app.get("/api/v1/jobs")
    async def list_jobs_endpoint(request: Request) -> JSONResponse:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("medical_data:read")
        except PermissionError as exc:
            return JSONResponse({"error": str(exc)}, status_code=403)

        records = jobs.list_jobs(principal.institution_id)
        return JSONResponse({"jobs": [r.to_dict() for r in records], "count": len(records)})

    @app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse)
    async def get_job_endpoint(job_id: str, request: Request) -> JSONResponse:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("medical_data:read")
        except PermissionError as exc:
            return JSONResponse({"error": str(exc)}, status_code=403)

        rec = jobs.get_job(job_id)
        if rec is None:
            return JSONResponse({"error": "Job not found"}, status_code=404)
        try:
            principal.assert_institution_access(rec.institution_id)
        except PermissionError as exc:
            return JSONResponse({"error": str(exc)}, status_code=403)
        return JSONResponse(rec.to_dict())

    @app.get("/api/v1/jobs/{job_id}/events")
    async def job_events_sse_endpoint(job_id: str, request: Request) -> Any:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("medical_data:read")
        except PermissionError as exc:
            return JSONResponse({"error": str(exc)}, status_code=403)

        rec = jobs.get_job(job_id)
        if rec is None:
            return JSONResponse({"error": "Job not found"}, status_code=404)
        try:
            principal.assert_institution_access(rec.institution_id)
        except PermissionError as exc:
            return JSONResponse({"error": str(exc)}, status_code=403)

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
            return JSONResponse({"error": str(exc)}, status_code=403)

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
        except (PermissionError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.get("/api/v1/subjects/{subject_id}/timeline")
    async def subject_timeline_endpoint(subject_id: str, request: Request) -> JSONResponse:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("medical_data:read")
        except PermissionError as exc:
            return JSONResponse({"error": str(exc)}, status_code=403)

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
            return JSONResponse({"error": str(exc)}, status_code=403)

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
        term: str = Query(""),
        category: str | None = Query(None),
    ) -> JSONResponse:
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
    async def list_models_endpoint() -> JSONResponse:
        service.ensure_models_trained_and_registered()
        with service.db.connect() as conn:
            rows = [dict(r) for r in conn.execute("SELECT * FROM models ORDER BY model_id ASC").fetchall()]
        return JSONResponse({"models": rows, "count": len(rows)})

    @app.get("/api/v1/experiments")
    async def list_experiments_endpoint() -> JSONResponse:
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
            return JSONResponse({"error": str(exc)}, status_code=403)

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
            return JSONResponse({"error": str(exc)}, status_code=403)

        from medresearch_platform.extraction.benchmark import run_six_approach_benchmark

        epochs = body.epochs if body is not None else 15
        seed = body.seed if body is not None else 42
        report = run_six_approach_benchmark(epochs=epochs, seed=seed)
        return JSONResponse(report)

    @app.get("/api/v1/audit/verify")
    async def verify_audit_endpoint(request: Request) -> JSONResponse:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("audit:read")
        except PermissionError as exc:
            return JSONResponse({"error": str(exc)}, status_code=403)

        with service.db.connect() as conn:
            result = verify_audit_chain(conn)
        return JSONResponse(result)

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

    return app
