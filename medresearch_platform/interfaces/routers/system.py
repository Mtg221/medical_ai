"""System, auth, schema, model provenance, and audit verification endpoints."""

from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from medresearch_platform import __spec_coverage__, __version__
from medresearch_platform.diagnostics import generate_readiness_report
from medresearch_platform.documents.ocr_engine import get_ocr_status
from medresearch_platform.extraction.schema_models import (
    CLINICAL_RESEARCH_NOTICE,
    ExtractionResultSchema,
)
from medresearch_platform.interfaces.routers.common import (
    auth_error_response,
    extract_principal,
)
from medresearch_platform.interfaces.schemas import TokenRequest, TokenResponse
from medresearch_platform.security.audit import verify_audit_chain
from medresearch_platform.security.auth import issue_access_token, verify_password
from medresearch_platform.workflows.pipeline import MedicalResearchPlatformService


def build_system_router(service: MedicalResearchPlatformService) -> APIRouter:
    router = APIRouter(tags=["system"])

    @router.get("/api/v1/health")
    async def health_endpoint() -> JSONResponse:
        report = generate_readiness_report(service.config.workspace_dir)
        vision_status = service.vision_engine.status()
        return JSONResponse(
            {
                "status": "ok",
                "version": __version__,
                "specification_coverage": __spec_coverage__,
                "ocr": get_ocr_status(),
                "vision_router": vision_status["vision_router"],
                "vision_skin": vision_status["vision_skin"],
                "readiness": report.to_dict(),
                "clinical_notice": CLINICAL_RESEARCH_NOTICE,
            }
        )

    @router.get("/api/v1/schema/extraction")
    async def extraction_schema_endpoint(request: Request) -> JSONResponse:
        try:
            extract_principal(request, service)
        except PermissionError as exc:
            return auth_error_response(exc)
        schema = ExtractionResultSchema.model_json_schema()
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        return JSONResponse(schema)

    @router.post("/api/v1/auth/token", response_model=TokenResponse)
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
            return JSONResponse(
                {
                    "access_token": token,
                    "token_type": "Bearer",
                    "user_id": str(row["user_id"]),
                    "institution_id": str(row["institution_id"]),
                    "role": str(row["role"]),
                    "permissions": perms,
                }
            )

    @router.get("/api/v1/models")
    async def list_models_endpoint(request: Request) -> JSONResponse:
        try:
            extract_principal(request, service)
        except PermissionError as exc:
            return auth_error_response(exc)

        service.ensure_models_trained_and_registered()
        with service.db.connect() as conn:
            rows = [dict(r) for r in conn.execute("SELECT * FROM models ORDER BY model_id ASC").fetchall()]
        loaded_models = service.extractor.list_loaded_models()
        return JSONResponse({"models": rows, "loaded_models": loaded_models, "count": len(rows)})

    @router.get("/api/v1/audit/verify")
    async def verify_audit_endpoint(request: Request) -> JSONResponse:
        try:
            principal = extract_principal(request, service)
            principal.assert_permission("audit:read")
        except PermissionError as exc:
            return auth_error_response(exc)

        with service.db.connect() as conn:
            result = verify_audit_chain(conn)
        return JSONResponse(result)

    return router
