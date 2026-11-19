"""Experiment tracking and extraction benchmark routes."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from medresearch_platform.interfaces.routers.common import (
    auth_error_response,
    extract_principal,
)
from medresearch_platform.interfaces.schemas import BenchmarkRequest
from medresearch_platform.workflows.pipeline import MedicalResearchPlatformService


def build_experiments_router(service: MedicalResearchPlatformService) -> APIRouter:
    router = APIRouter(tags=["experiments"])

    @router.get("/api/v1/experiments")
    async def list_experiments_endpoint(request: Request) -> JSONResponse:
        try:
            extract_principal(request, service)
        except PermissionError as exc:
            return auth_error_response(exc)

        with service.db.connect() as conn:
            exps = [dict(r) for r in conn.execute("SELECT * FROM experiments ORDER BY created_at DESC").fetchall()]
            rounds = [
                dict(r)
                for r in conn.execute("SELECT * FROM federated_rounds ORDER BY round_number ASC").fetchall()
            ]
        return JSONResponse({"experiments": exps, "federated_rounds": rounds})

    @router.post("/api/v1/experiments/run")
    async def run_experiments_endpoint(request: Request) -> JSONResponse:
        try:
            principal = extract_principal(request, service)
            principal.assert_permission("training:execute")
        except PermissionError as exc:
            return auth_error_response(exc)

        manifest = service.run_phase1_experiments()
        return JSONResponse(manifest)

    @router.post("/api/v1/extraction/benchmark")
    async def run_six_extraction_benchmark_endpoint(
        request: Request,
        body: BenchmarkRequest | None = None,
    ) -> JSONResponse:
        try:
            principal = extract_principal(request, service)
            principal.assert_permission("training:execute")
        except PermissionError as exc:
            return auth_error_response(exc)

        epochs = body.epochs if body is not None else 15
        seed = body.seed if body is not None else 42
        v3_res = Path("docs/results/v3_locked_test_evaluation.json")
        if v3_res.exists():
            payload = json.loads(v3_res.read_text(encoding="utf-8"))
            return JSONResponse({"source": str(v3_res), "epochs": epochs, "seed": seed, "results": payload})
        return JSONResponse({"source": "none", "epochs": epochs, "seed": seed, "results": {}})

    return router
