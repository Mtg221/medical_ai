"""Starlette ASGI REST API, OpenAPI 3.1 schema, and Accessible Bilingual (EN/FR) Web Workspace."""

from __future__ import annotations

import base64
import json
from typing import Any
from starlette.applications import Starlette
from starlette.requests import Request
from pathlib import Path
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from medresearch_platform import __spec_coverage__, __version__
from medresearch_platform.config import PlatformConfig
from medresearch_platform.diagnostics import generate_readiness_report
from medresearch_platform.security.audit import verify_audit_chain
from medresearch_platform.security.auth import (
    AuthenticatedPrincipal,
    issue_access_token,
    verify_access_token,
    verify_password,
)
from medresearch_platform.workflows.pipeline import MedicalResearchPlatformService
from medresearch_platform.workflows.review import submit_entity_human_review


STATIC_DIR = Path(__file__).resolve().parent / "static"


def _extract_principal(request: Request, service: MedicalResearchPlatformService) -> AuthenticatedPrincipal:
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[len("Bearer ") :].strip()
        return verify_access_token(token, service.config.token_secret_key)
    raise PermissionError("Missing or invalid Authorization: Bearer <token> header.")


def create_app(config: PlatformConfig | None = None) -> Starlette:
    """Create the Starlette ASGI application for the medical research platform."""
    service = MedicalResearchPlatformService(config=config)
    service.initialize_platform()

    async def health_endpoint(request: Request) -> JSONResponse:
        report = generate_readiness_report(service.config.workspace_dir)
        return JSONResponse({
            "status": "ok",
            "version": __version__,
            "specification_coverage": __spec_coverage__,
            "readiness": report.to_dict(),
        })

    async def openapi_endpoint(request: Request) -> JSONResponse:
        return JSONResponse({
            "openapi": "3.1.0",
            "info": {
                "title": "Medical Information Intelligence & Federated Scientific Research Platform API",
                "version": __version__,
                "description": (
                    "Research-grade bilingual (EN/FR) medical document extraction, custom random-initialized "
                    "neural models, longitudinal knowledge graph, and federated research API. "
                    "NOT FOR AUTONOMOUS CLINICAL DIAGNOSIS OR TREATMENT."
                ),
            },
            "paths": {
                "/api/v1/health": {"get": {"summary": "System readiness and hardware diagnostics"}},
                "/api/v1/auth/token": {"post": {"summary": "Issue scoped HMAC access token"}},
                "/api/v1/documents": {"get": {"summary": "List institution-scoped documents"}},
                "/api/v1/documents/{document_id}": {"get": {"summary": "Inspect document, OCR blocks, entities, and relations"}},
                "/api/v1/documents/ingest": {"post": {"summary": "Ingest and extract an authorized document"}},
                "/api/v1/entities/{entity_id}/review": {"post": {"summary": "Submit human review or correction"}},
                "/api/v1/subjects/{subject_id}/timeline": {"get": {"summary": "Retrieve longitudinal subject timeline and contradictions"}},
                "/api/v1/search": {"get": {"summary": "Permission-aware hybrid lexical, structured, and neural search"}},
                "/api/v1/terminology/map": {"get": {"summary": "Map English or French clinical mention to controlled vocabulary"}},
                "/api/v1/models": {"get": {"summary": "List registered custom models and random-init verification hashes"}},
                "/api/v1/experiments": {"get": {"summary": "List experiment manifests, calibration, fairness, and federated rounds"}},
                "/api/v1/experiments/run": {"post": {"summary": "Execute Phase 1 CPU-only reproducible benchmark"}},
                "/api/v1/audit/verify": {"get": {"summary": "Verify tamper-evident SHA-256 audit log chain"}},
            },
        })

    async def issue_token_endpoint(request: Request) -> JSONResponse:
        body = await request.json()
        username = str(body.get("username", ""))
        password = str(body.get("password", ""))

        with service.db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ? AND is_active = 1",
                (username,),
            ).fetchone()
            if row is None or not verify_password(str(row["password_hash"]), password):
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

    async def get_document_detail_endpoint(request: Request) -> JSONResponse:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("medical_data:read")
        except PermissionError as exc:
            return JSONResponse({"error": str(exc)}, status_code=403)

        doc_id = request.path_params["document_id"]
        with service.db.connect() as conn:
            doc = conn.execute("SELECT * FROM documents WHERE document_id = ?", (doc_id,)).fetchone()
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
                    (doc_id,),
                ).fetchall()
            ]
            sections = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM document_sections WHERE document_id = ? ORDER BY page_number ASC, char_start ASC",
                    (doc_id,),
                ).fetchall()
            ]
            ocr_blocks = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM ocr_blocks WHERE document_id = ? ORDER BY block_index ASC",
                    (doc_id,),
                ).fetchall()
            ]
            entities = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM extracted_entities WHERE document_id = ? ORDER BY char_start ASC",
                    (doc_id,),
                ).fetchall()
            ]
            relations = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM extracted_relations WHERE document_id = ?",
                    (doc_id,),
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

    async def ingest_document_endpoint(request: Request) -> JSONResponse:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("medical_data:write")
        except PermissionError as exc:
            return JSONResponse({"error": str(exc)}, status_code=403)

        body = await request.json()
        filename = str(body.get("filename", "uploaded_note.txt"))
        if "content_base64" in body:
            raw_bytes = base64.b64decode(str(body["content_base64"]))
        else:
            raw_bytes = str(body.get("text", "")).encode("utf-8")

        try:
            result = service.ingest_document(
                principal=principal,
                raw_bytes=raw_bytes,
                filename=filename,
                dataset_id=str(body.get("dataset_id", "ds_synth_bilingual_v1")),
                subject_raw_id=str(body.get("subject_id", "subj_synth_001")),
                family_id=str(body.get("family_id", "fam_synth_upload")),
                split_assignment=str(body.get("split_assignment", "unassigned")),
                encounter_date_iso=str(body.get("encounter_date_iso", "2026-06-01")),
                specialty=str(body.get("specialty", "INTERNAL_MEDICINE")),
                document_type=str(body.get("document_type", "CLINICAL_NOTE")),
            )
            return JSONResponse(result, status_code=201)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    async def review_entity_endpoint(request: Request) -> JSONResponse:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("annotation:review")
        except PermissionError as exc:
            return JSONResponse({"error": str(exc)}, status_code=403)

        entity_id = request.path_params["entity_id"]
        body = await request.json()
        try:
            with service.db.transaction() as conn:
                res = submit_entity_human_review(
                    conn,
                    principal=principal,
                    entity_id=entity_id,
                    review_decision=str(body.get("review_decision", "CORRECT_AND_APPROVE")),
                    corrected_entity_type=body.get("corrected_entity_type"),
                    corrected_assertion_status=body.get("corrected_assertion_status"),
                    corrected_temporal_status=body.get("corrected_temporal_status"),
                    corrected_normalized_text=body.get("corrected_normalized_text"),
                    rationale=str(body.get("rationale", "Clinical annotator review")),
                )
            return JSONResponse(res)
        except KeyError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        except (PermissionError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    async def subject_timeline_endpoint(request: Request) -> JSONResponse:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("medical_data:read")
        except PermissionError as exc:
            return JSONResponse({"error": str(exc)}, status_code=403)

        subject_id = request.path_params["subject_id"]
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

    async def search_endpoint(request: Request) -> JSONResponse:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("search:query")
        except PermissionError as exc:
            return JSONResponse({"error": str(exc)}, status_code=403)

        query = request.query_params.get("q", "").strip()
        if not query:
            return JSONResponse({"error": "Query parameter 'q' is required."}, status_code=400)

        with service.db.connect() as conn:
            hits = service.retrieval_model.search_documents(
                conn,
                institution_id=principal.institution_id,
                query=query,
                top_k=int(request.query_params.get("top_k", "5")),
            )
        return JSONResponse({"query": query, "results": hits, "count": len(hits)})

    async def terminology_map_endpoint(request: Request) -> JSONResponse:
        term = request.query_params.get("term", "").strip()
        category = request.query_params.get("category")
        res = service.terminology.map_mention(term, category_hint=category)
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

    async def list_models_endpoint(request: Request) -> JSONResponse:
        service.ensure_models_trained_and_registered()
        with service.db.connect() as conn:
            rows = [dict(r) for r in conn.execute("SELECT * FROM models ORDER BY model_id ASC").fetchall()]
        return JSONResponse({"models": rows, "count": len(rows)})

    async def list_experiments_endpoint(request: Request) -> JSONResponse:
        with service.db.connect() as conn:
            exps = [dict(r) for r in conn.execute("SELECT * FROM experiments ORDER BY created_at DESC").fetchall()]
            rounds = [dict(r) for r in conn.execute("SELECT * FROM federated_rounds ORDER BY round_number ASC").fetchall()]
        return JSONResponse({"experiments": exps, "federated_rounds": rounds})

    async def run_experiments_endpoint(request: Request) -> JSONResponse:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("training:execute")
        except PermissionError as exc:
            return JSONResponse({"error": str(exc)}, status_code=403)

        manifest = service.run_phase1_experiments()
        return JSONResponse(manifest)

    async def run_six_extraction_benchmark_endpoint(request: Request) -> JSONResponse:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("training:execute")
        except PermissionError as exc:
            return JSONResponse({"error": str(exc)}, status_code=403)

        from medresearch_platform.extraction.benchmark import run_six_approach_benchmark

        try:
            body = await request.json()
        except Exception:
            body = {}
        epochs = max(5, min(40, int(body.get("epochs", 15)))) if isinstance(body, dict) else 15
        seed = int(body.get("seed", 42)) if isinstance(body, dict) else 42
        report = run_six_approach_benchmark(epochs=epochs, seed=seed)
        return JSONResponse(report)

    async def run_derm_benchmark_endpoint(request: Request) -> JSONResponse:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("training:execute")
        except PermissionError as exc:
            return JSONResponse({"error": str(exc)}, status_code=403)

        from medresearch_platform.dermatology.pipeline import run_derm_benchmark

        try:
            body = await request.json()
        except Exception:
            body = {}
        epochs = max(5, min(40, int(body.get("epochs", 15)))) if isinstance(body, dict) else 15
        per_class = max(10, min(300, int(body.get("per_class_limit", 150)))) if isinstance(body, dict) else 150
        seed = int(body.get("seed", 42)) if isinstance(body, dict) else 42
        report = run_derm_benchmark(per_class_limit=per_class, epochs=epochs, seed=seed)
        return JSONResponse(report)

    async def analyze_vision_endpoint(request: Request) -> JSONResponse:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("medical_data:read")
        except PermissionError as exc:
            return JSONResponse({"error": str(exc)}, status_code=403)

        import io
        import numpy as np
        from PIL import Image
        from medresearch_platform.dermatology.catalog import MALIGNANCY_CLASSES
        from medresearch_platform.dermatology.vision_classifier import RandomInitDermConvNet
        from medresearch_platform.extraction.nn_utils import softmax

        body = await request.json()
        raw_b64 = str(body.get("content_base64", ""))
        if not raw_b64:
            return JSONResponse({"error": "content_base64 is required"}, status_code=400)
        try:
            raw_bytes = base64.b64decode(raw_b64)
            with Image.open(io.BytesIO(raw_bytes)) as img:
                rgb_img = img.convert("RGB")
                vision_report = service.vision_model.analyze_image(rgb_img)
                arr = np.asarray(rgb_img.resize((32, 32)), dtype=np.float64) / 255.0
        except Exception as exc:
            return JSONResponse({"error": f"Invalid image payload: {exc}"}, status_code=400)

        derm_net = RandomInitDermConvNet(seed=service.config.default_random_seed)
        logits, _ = derm_net.forward(arr[None, ...])
        probs = softmax(logits[0], axis=-1)
        pred_idx = int(np.argmax(probs))

        return JSONResponse({
            "vision_model": vision_report.model_id,
            "width": vision_report.width,
            "height": vision_report.height,
            "quality_score": vision_report.quality_score,
            "predicted_modality": vision_report.predicted_modality,
            "roi_bbox": vision_report.roi_bbox,
            "vision_confidence": vision_report.confidence,
            "derm_model": f"{derm_net.architecture_name}-{derm_net.architecture_version}",
            "derm_predicted_group": MALIGNANCY_CLASSES[pred_idx],
            "derm_probabilities": {
                cls_name: round(float(probs[i]), 4) for i, cls_name in enumerate(MALIGNANCY_CLASSES)
            },
            "pretrained_weights_used": False,
            "disclaimer": "Research-only random-initialized model output; NOT a clinical diagnosis.",
        })

    async def verify_audit_endpoint(request: Request) -> JSONResponse:
        try:
            principal = _extract_principal(request, service)
            principal.assert_permission("audit:read")
        except PermissionError as exc:
            return JSONResponse({"error": str(exc)}, status_code=403)

        with service.db.connect() as conn:
            result = verify_audit_chain(conn)
        return JSONResponse(result)

    async def web_workspace_endpoint(request: Request) -> FileResponse:
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

    REACT_DIST_DIR = Path(__file__).resolve().parents[2] / "frontend" / "dist"

    async def react_workspace_endpoint(request: Request) -> FileResponse:
        """Serve the React + Vite workspace bundle if built."""
        index_file = REACT_DIST_DIR / "index.html"
        if not index_file.exists():
            return JSONResponse({"error": "React frontend not built. Run: npm --prefix frontend run build"}, status_code=404)
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

    routes = [
        Route("/", web_workspace_endpoint, methods=["GET"]),
        Route("/react", react_workspace_endpoint, methods=["GET"]),
        Route("/react/", react_workspace_endpoint, methods=["GET"]),
        Route("/api/v1/health", health_endpoint, methods=["GET"]),
        Route("/api/v1/openapi.json", openapi_endpoint, methods=["GET"]),
        Route("/api/v1/auth/token", issue_token_endpoint, methods=["POST"]),
        Route("/api/v1/documents", list_documents_endpoint, methods=["GET"]),
        Route("/api/v1/documents/ingest", ingest_document_endpoint, methods=["POST"]),
        Route("/api/v1/documents/{document_id}", get_document_detail_endpoint, methods=["GET"]),
        Route("/api/v1/entities/{entity_id}/review", review_entity_endpoint, methods=["POST"]),
        Route("/api/v1/subjects/{subject_id}/timeline", subject_timeline_endpoint, methods=["GET"]),
        Route("/api/v1/search", search_endpoint, methods=["GET"]),
        Route("/api/v1/terminology/map", terminology_map_endpoint, methods=["GET"]),
        Route("/api/v1/models", list_models_endpoint, methods=["GET"]),
        Route("/api/v1/experiments", list_experiments_endpoint, methods=["GET"]),
        Route("/api/v1/experiments/run", run_experiments_endpoint, methods=["POST"]),
        Route("/api/v1/extraction/benchmark", run_six_extraction_benchmark_endpoint, methods=["POST"]),
        Route("/api/v1/dermatology/benchmark", run_derm_benchmark_endpoint, methods=["POST"]),
        Route("/api/v1/vision/analyze", analyze_vision_endpoint, methods=["POST"]),
        Route("/api/v1/audit/verify", verify_audit_endpoint, methods=["GET"]),
        Mount("/static", app=StaticFiles(directory=str(STATIC_DIR)), name="static"),
    ]
    if (REACT_DIST_DIR / "assets").exists():
        routes.append(Mount("/react/assets", app=StaticFiles(directory=str(REACT_DIST_DIR / "assets")), name="react_assets"))

    app = Starlette(debug=(service.config.profile == "development"), routes=routes)
    app.state.service = service
    return app
