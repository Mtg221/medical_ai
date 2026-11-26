"""Typed Pydantic v2 request and response schemas for the FastAPI REST API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    user_id: str
    institution_id: str
    role: str
    permissions: list[str]


class IngestDocumentRequest(BaseModel):
    filename: str = "uploaded_note.txt"
    text: str = ""
    content_base64: str | None = None
    dataset_id: str = "ds_synth_bilingual_v1"
    subject_id: str = ""
    family_id: str = "fam_synth_upload"
    split_assignment: str = "unassigned"
    encounter_date_iso: str = "2026-06-01"
    specialty: str = "INTERNAL_MEDICINE"
    document_type: str = "CLINICAL_NOTE"
    force_new: bool = False


class ReviewEntityRequest(BaseModel):
    review_decision: str = "CORRECT_AND_APPROVE"
    corrected_entity_type: str | None = None
    corrected_assertion_status: str | None = None
    corrected_certainty: str | None = None
    corrected_temporality: str | None = None
    corrected_experiencer: str | None = None
    corrected_temporal_status: str | None = None
    corrected_normalized_text: str | None = None
    rationale: str = "Clinical annotator review"


class BenchmarkRequest(BaseModel):
    epochs: int = Field(default=15, ge=5, le=40)
    seed: int = 42


class JobStatusResponse(BaseModel):
    job_id: str
    institution_id: str
    actor_user_id: str
    job_type: str
    original_filename: str = ""
    status: Literal["PENDING", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED"]
    progress_pct: int = Field(ge=0, le=100)
    stage: str = ""
    current_stage: str
    result: dict[str, Any] | None = None
    error: str | None = None
    error_message: str | None = None
    created_at: str
    updated_at: str
