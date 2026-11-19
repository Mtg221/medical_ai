"""Document ingestion, upload jobs/SSE, reports, timelines, search, and terminology routes."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, Form, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse

from medresearch_platform.documents.parsers import (
    PayloadTooLargeError,
    UnsupportedFileTypeError,
    validate_upload_payload,
)
from medresearch_platform.interfaces.routers.common import (
    auth_error_response,
    extract_principal,
)
from medresearch_platform.interfaces.schemas import IngestDocumentRequest, JobStatusResponse
from medresearch_platform.reports import ReportFormat, ReportLanguage, generate_report_bytes
from medresearch_platform.security.privacy import deidentify_clinical_text
from medresearch_platform.vision.serving import _row_to_image_finding_dict
from medresearch_platform.workflows.jobs import JobManager, execute_document_upload_job
from medresearch_platform.workflows.pipeline import MedicalResearchPlatformService


def build_documents_router(
    service: MedicalResearchPlatformService,
    jobs: JobManager,
) -> APIRouter:
    router = APIRouter(tags=["documents"])

    @router.get("/api/v1/documents")
    async def list_documents_endpoint(request: Request) -> JSONResponse:
        try:
            principal = extract_principal(request, service)
            principal.assert_permission("medical_data:read")
        except PermissionError as exc:
            return auth_error_response(exc)

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

    @router.get("/api/v1/documents/{document_id}")
    async def get_document_detail_endpoint(document_id: str, request: Request) -> JSONResponse:
        try:
            principal = extract_principal(request, service)
            principal.assert_permission("medical_data:read")
        except PermissionError as exc:
            return auth_error_response(exc)

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

        return JSONResponse(
            {
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
            }
        )

    @router.get("/api/v1/documents/{document_id}/report")
    async def document_report_endpoint(
        document_id: str,
        request: Request,
        format: ReportFormat = Query("markdown"),
        lang: ReportLanguage = Query("en"),
    ) -> Response:
        try:
            principal = extract_principal(request, service)
            principal.assert_permission("medical_data:read")
        except PermissionError as exc:
            return auth_error_response(exc)

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
            pages = conn.execute(
                "SELECT raw_text FROM document_pages WHERE document_id = ? ORDER BY page_number ASC",
                (document_id,),
            ).fetchall()
            full_raw = "\n".join(str(p["raw_text"]) for p in pages)
            deid = deidentify_clinical_text(full_raw)
            doc_dict = dict(doc)
            doc_dict["pii_spans"] = [
                {"original": s.original_value, "token": s.replacement_token}
                for s in deid.detected_spans
            ]

        payload, media_type = generate_report_bytes(
            doc_dict,
            entities,
            relations,
            image_findings=image_findings,
            same_lesion_links=same_lesion_links,
            lang=lang,
            fmt=format,
        )
        return Response(content=payload, media_type=media_type)

    @router.post("/api/v1/documents/ingest")
    async def ingest_document_endpoint(body: IngestDocumentRequest, request: Request) -> JSONResponse:
        try:
            principal = extract_principal(request, service)
            principal.assert_permission("medical_data:write")
        except PermissionError as exc:
            return auth_error_response(exc)

        raw_bytes = (
            base64.b64decode(body.content_base64)
            if body.content_base64 is not None
            else body.text.encode("utf-8")
        )

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

    @router.post("/api/v1/documents/upload", response_model=JobStatusResponse, status_code=202)
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
            principal = extract_principal(request, service)
            principal.assert_permission("medical_data:write")
        except PermissionError as exc:
            return auth_error_response(exc)

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

    @router.get("/api/v1/jobs")
    async def list_jobs_endpoint(request: Request) -> JSONResponse:
        try:
            principal = extract_principal(request, service)
            principal.assert_permission("medical_data:read")
        except PermissionError as exc:
            return auth_error_response(exc)

        records = jobs.list_jobs(principal.institution_id)
        return JSONResponse({"jobs": [r.to_dict() for r in records], "count": len(records)})

    @router.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse)
    async def get_job_endpoint(job_id: str, request: Request) -> JSONResponse:
        try:
            principal = extract_principal(request, service)
            principal.assert_permission("medical_data:read")
        except PermissionError as exc:
            return auth_error_response(exc)

        rec = jobs.get_job(job_id)
        if rec is None:
            return JSONResponse({"error": "Job not found"}, status_code=404)
        try:
            principal.assert_institution_access(rec.institution_id)
        except PermissionError as exc:
            return auth_error_response(exc)
        return JSONResponse(rec.to_dict())

    @router.get("/api/v1/jobs/{job_id}/events")
    async def job_events_sse_endpoint(job_id: str, request: Request) -> Any:
        try:
            principal = extract_principal(request, service)
            principal.assert_permission("medical_data:read")
        except PermissionError as exc:
            return auth_error_response(exc)

        rec = jobs.get_job(job_id)
        if rec is None:
            return JSONResponse({"error": "Job not found"}, status_code=404)
        try:
            principal.assert_institution_access(rec.institution_id)
        except PermissionError as exc:
            return auth_error_response(exc)

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

    @router.get("/api/v1/subjects/{subject_id}/timeline")
    async def subject_timeline_endpoint(subject_id: str, request: Request) -> JSONResponse:
        try:
            principal = extract_principal(request, service)
            principal.assert_permission("medical_data:read")
        except PermissionError as exc:
            return auth_error_response(exc)

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
        return JSONResponse(
            {
                "subject": dict(subj),
                "timeline_events": events,
                "contradictions_count": sum(1 for e in events if int(e["contradiction_flag"]) == 1),
            }
        )

    @router.get("/api/v1/search")
    async def search_endpoint(
        request: Request,
        q: str = Query(""),
        top_k: int = Query(5, ge=1, le=50),
    ) -> JSONResponse:
        try:
            principal = extract_principal(request, service)
            principal.assert_permission("search:query")
        except PermissionError as exc:
            return auth_error_response(exc)

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

    @router.get("/api/v1/terminology/map")
    async def terminology_map_endpoint(
        request: Request,
        term: str = Query(""),
        category: str | None = Query(None),
    ) -> JSONResponse:
        try:
            principal = extract_principal(request, service)
            principal.assert_permission("terminology:read")
        except PermissionError as exc:
            return auth_error_response(exc)

        res = service.terminology.map_mention(term.strip(), category_hint=category)
        return JSONResponse(
            {
                "source_text": res.source_text,
                "normalized_text": res.normalized_text,
                "concept_id": res.concept_id,
                "vocabulary": res.vocabulary,
                "code": res.code,
                "match_type": res.match_type,
                "confidence": res.confidence,
                "candidate_codes": list(res.candidate_codes),
                "disclaimer": "A mapped terminology code is a normalized research concept, not an approved clinical diagnosis.",
            }
        )

    return router
