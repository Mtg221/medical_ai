"""FastAPI ASGI REST API factory wiring modular domain routers, auth middleware, and web workspaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from medresearch_platform import __version__
from medresearch_platform.config import PlatformConfig
from medresearch_platform.extraction.schema_models import CLINICAL_RESEARCH_NOTICE
from medresearch_platform.interfaces.routers import (
    ImageReviewRequest,
    ReviewSubmissionRequest,
    auth_error_response,
    build_documents_router,
    build_experiments_router,
    build_reviews_router,
    build_system_router,
    build_vision_router,
    extract_principal,
)
from medresearch_platform.workflows.jobs import JobManager
from medresearch_platform.workflows.pipeline import MedicalResearchPlatformService

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Backward-compatible private helper aliases
_extract_principal = extract_principal
_auth_error_response = auth_error_response

__all__ = [
    "ImageReviewRequest",
    "ReviewSubmissionRequest",
    "STATIC_DIR",
    "_auth_error_response",
    "_extract_principal",
    "create_app",
]


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
                extract_principal(request, service)
            except PermissionError as exc:
                return auth_error_response(exc)
        return await call_next(request)

    app.include_router(build_system_router(service))
    app.include_router(build_documents_router(service, jobs))
    app.include_router(build_reviews_router(service))
    app.include_router(build_experiments_router(service))
    app.include_router(build_vision_router(service, jobs))

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
                "X-Workspace-UI": "legacy-fallback; canonical=/react/",
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
                "X-Workspace-UI": "react-canonical",
            },
        )

    app.add_api_route("/react", react_workspace_endpoint, methods=["GET"], response_model=None)
    app.add_api_route("/react/", react_workspace_endpoint, methods=["GET"], response_model=None)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    if (react_dist_dir / "assets").exists():
        app.mount("/react/assets", StaticFiles(directory=str(react_dist_dir / "assets")), name="react_assets")
    app.add_api_route("/react/{full_path:path}", react_workspace_endpoint, methods=["GET"], response_model=None)

    return app
