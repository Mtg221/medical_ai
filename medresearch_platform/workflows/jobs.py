"""SQLite-backed background job manager with stage progress tracking for FastAPI."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from medresearch_platform.db.engine import DatabaseEngine
from medresearch_platform.security.auth import AuthenticatedPrincipal
from medresearch_platform.workflows.pipeline import MedicalResearchPlatformService

JobStatus = Literal["PENDING", "RUNNING", "SUCCEEDED", "FAILED"]


@dataclass(frozen=True)
class JobRecord:
    """Persisted background job state."""

    job_id: str
    institution_id: str
    actor_user_id: str
    job_type: str
    status: JobStatus
    progress_pct: int
    current_stage: str
    result: dict[str, Any] | None
    error_message: str | None
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "institution_id": self.institution_id,
            "actor_user_id": self.actor_user_id,
            "job_type": self.job_type,
            "status": self.status,
            "progress_pct": self.progress_pct,
            "current_stage": self.current_stage,
            "result": self.result,
            "error_message": self.error_message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class JobManager:
    """Manages SQLite-persisted background jobs for asynchronous document ingestion & extraction."""

    def __init__(self, db: DatabaseEngine) -> None:
        self.db = db

    def create_job(
        self,
        *,
        institution_id: str,
        actor_user_id: str,
        job_type: str,
        initial_stage: str = "QUEUED",
    ) -> JobRecord:
        now = datetime.now(timezone.utc).isoformat()
        job_id = f"job_{uuid.uuid4().hex[:14]}"
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, institution_id, actor_user_id, job_type, status,
                    progress_pct, current_stage, result_json, error_message,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'PENDING', 0, ?, NULL, NULL, ?, ?)
                """,
                (job_id, institution_id, actor_user_id, job_type, initial_stage, now, now),
            )
        return JobRecord(
            job_id=job_id,
            institution_id=institution_id,
            actor_user_id=actor_user_id,
            job_type=job_type,
            status="PENDING",
            progress_pct=0,
            current_stage=initial_stage,
            result=None,
            error_message=None,
            created_at=now,
            updated_at=now,
        )

    def update_job(
        self,
        job_id: str,
        *,
        status: JobStatus,
        progress_pct: int,
        current_stage: str,
        result: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        result_json = json.dumps(result) if result is not None else None
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, progress_pct = ?, current_stage = ?,
                    result_json = COALESCE(?, result_json),
                    error_message = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (
                    status,
                    max(0, min(100, progress_pct)),
                    current_stage,
                    result_json,
                    error_message,
                    now,
                    job_id,
                ),
            )

    def get_job(self, job_id: str) -> JobRecord | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                return None
            return self._row_to_record(row)

    def list_jobs(self, institution_id: str, limit: int = 50) -> list[JobRecord]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE institution_id = ? ORDER BY created_at DESC LIMIT ?",
                (institution_id, limit),
            ).fetchall()
            return [self._row_to_record(r) for r in rows]

    @staticmethod
    def _row_to_record(row: Any) -> JobRecord:
        res_raw = row["result_json"]
        res = json.loads(str(res_raw)) if res_raw else None
        return JobRecord(
            job_id=str(row["job_id"]),
            institution_id=str(row["institution_id"]),
            actor_user_id=str(row["actor_user_id"]),
            job_type=str(row["job_type"]),
            status=str(row["status"]),  # type: ignore[arg-type]
            progress_pct=int(row["progress_pct"]),
            current_stage=str(row["current_stage"]),
            result=res,
            error_message=str(row["error_message"]) if row["error_message"] else None,
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )


def execute_document_upload_job(
    service: MedicalResearchPlatformService,
    jobs: JobManager,
    *,
    job_id: str,
    principal: AuthenticatedPrincipal,
    raw_bytes: bytes,
    filename: str,
    dataset_id: str,
    subject_id: str,
    family_id: str,
    split_assignment: str,
    encounter_date_iso: str,
    specialty: str,
    document_type: str,
) -> None:
    """Background worker that parses/OCRs, extracts, normalizes, and persists an uploaded document."""
    try:
        jobs.update_job(job_id, status="RUNNING", progress_pct=20, current_stage="PARSING_AND_OCR")
        result = service.ingest_document(
            principal=principal,
            raw_bytes=raw_bytes,
            filename=filename,
            dataset_id=dataset_id,
            subject_raw_id=subject_id,
            family_id=family_id,
            split_assignment=split_assignment,
            encounter_date_iso=encounter_date_iso,
            specialty=specialty,
            document_type=document_type,
        )
        jobs.update_job(
            job_id,
            status="SUCCEEDED",
            progress_pct=100,
            current_stage="COMPLETED",
            result=result,
        )
    except Exception as exc:
        jobs.update_job(
            job_id,
            status="FAILED",
            progress_pct=100,
            current_stage="FAILED",
            error_message=str(exc),
        )
