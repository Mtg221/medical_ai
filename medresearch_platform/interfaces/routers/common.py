"""Shared request schemas and Bearer authentication helpers for API routers."""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from medresearch_platform.security.auth import (
    AuthenticatedPrincipal,
    AuthenticationError,
    verify_access_token,
)
from medresearch_platform.workflows.pipeline import MedicalResearchPlatformService


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


def extract_principal(
    request: Request,
    service: MedicalResearchPlatformService,
) -> AuthenticatedPrincipal:
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[len("Bearer ") :].strip()
        if token:
            return verify_access_token(token, service.config.token_secret_key)
    raise AuthenticationError("Missing or invalid Authorization: Bearer <token> header.")


def auth_error_response(exc: PermissionError) -> JSONResponse:
    if isinstance(exc, AuthenticationError):
        return JSONResponse(
            {"error": str(exc)},
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )
    return JSONResponse({"error": str(exc)}, status_code=403)
