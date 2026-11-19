"""Human clinical review submission, listing, and gold-standard export endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from medresearch_platform.interfaces.routers.common import (
    ReviewSubmissionRequest,
    auth_error_response,
    extract_principal,
)
from medresearch_platform.interfaces.schemas import ReviewEntityRequest
from medresearch_platform.security.auth import AuthenticationError
from medresearch_platform.workflows.pipeline import MedicalResearchPlatformService
from medresearch_platform.workflows.review import (
    export_reviewed_gold,
    list_human_reviews,
    submit_entity_human_review,
)


def build_reviews_router(service: MedicalResearchPlatformService) -> APIRouter:
    router = APIRouter(tags=["reviews"])

    @router.post("/api/v1/entities/{entity_id}/review")
    async def review_entity_endpoint(
        entity_id: str,
        body: ReviewEntityRequest,
        request: Request,
    ) -> JSONResponse:
        try:
            principal = extract_principal(request, service)
            principal.assert_permission("annotation:review")
        except PermissionError as exc:
            return auth_error_response(exc)

        decision_aliases = {
            "approve": "APPROVE_AS_GROUND_TRUTH",
            "correct": "CORRECT_AND_APPROVE",
            "flag": "FLAG_AMBIGUOUS",
            "reject": "REJECT",
        }
        normalized_decision = decision_aliases.get(
            body.review_decision.lower().strip(), body.review_decision
        )
        try:
            with service.db.transaction() as conn:
                res = submit_entity_human_review(
                    conn,
                    principal=principal,
                    entity_id=entity_id,
                    review_decision=normalized_decision,
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
            return auth_error_response(exc)
        except (PermissionError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @router.post("/api/v1/reviews")
    async def submit_review_endpoint(
        body: ReviewSubmissionRequest,
        request: Request,
    ) -> JSONResponse:
        try:
            principal = extract_principal(request, service)
            principal.assert_permission("annotation:review")
        except PermissionError as exc:
            return auth_error_response(exc)

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
            return auth_error_response(exc)
        except (PermissionError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @router.get("/api/v1/reviews")
    async def list_reviews_endpoint(
        request: Request,
        document_id: str | None = Query(default=None),
    ) -> JSONResponse:
        try:
            principal = extract_principal(request, service)
            principal.assert_permission("medical_data:read")
        except PermissionError as exc:
            return auth_error_response(exc)

        with service.db.connect() as conn:
            reviews = list_human_reviews(
                conn,
                institution_id=principal.institution_id,
                document_id=document_id,
            )
        return JSONResponse({"reviews": reviews, "count": len(reviews)})

    @router.get("/api/v1/reviews/export")
    async def export_reviews_endpoint(
        request: Request,
        format: str = Query(default="jsonl"),
        document_id: str | None = Query(default=None),
    ) -> Response:
        try:
            principal = extract_principal(request, service)
            principal.assert_permission("medical_data:read")
        except PermissionError as exc:
            return auth_error_response(exc)

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

    return router
