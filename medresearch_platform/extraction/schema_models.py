"""Pydantic v2 single-source-of-truth extraction contract and JSON Schema exporter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

AssertionStatusType = Literal[
    "present", "absent", "uncertain", "hypothetical", "family_history", "historical"
]
CertaintyType = Literal["confirmed", "possible", "absent"]
TemporalityType = Literal["current", "historical"]
ExperiencerType = Literal["patient", "family"]
NegationStatusType = Literal["affirmed", "negated", "uncertain"]
ReviewStatusType = Literal[
    "unreviewed", "needs_review", "approved", "corrected", "rejected", "flagged", "image_unusable"
]
RiskBandType = Literal["low", "medium", "high", "abstained"]

CLINICAL_RESEARCH_NOTICE = (
    "Research tool only — NOT for clinical diagnosis or treatment. "
    "Human clinician review required before any clinical use."
)
DERM_SAFETY_NOTICE_EN = "Research tool — not a diagnosis. A dermatologist must examine the lesion."
DERM_SAFETY_NOTICE_FR = "Outil de recherche — ne constitue pas un diagnostic. Un dermatologue doit examiner la lésion."


class DiagnosisCandidateSchema(BaseModel):
    """Top-K differential diagnosis candidate with calibrated probability."""

    rank: int = Field(ge=1)
    diagnosis: str
    probability: float = Field(ge=0.0, le=1.0)


class ImageFindingSchema(BaseModel):
    """Structured output contract for a dermatology/clinical image finding (Step 4)."""

    image_id: str
    document_id: str | None = None
    patient_id: str | None = None
    picture_type: str
    quality_result: dict[str, Any] = Field(default_factory=dict)
    malignancy_risk_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    malignancy_risk_band: RiskBandType = "abstained"
    top3_diagnoses: list[DiagnosisCandidateSchema] = Field(default_factory=list)
    heatmap_uri: str | None = None
    abstained: bool = False
    abstain_reason_en: str | None = None
    abstain_reason_fr: str | None = None
    body_site: str | None = None
    captured_date_iso: str | None = None
    model_name: str
    model_version: str
    git_commit: str = ""
    review_status: ReviewStatusType = "needs_review"
    provenance: dict[str, Any] = Field(default_factory=dict)
    safety_notice_en: str = DERM_SAFETY_NOTICE_EN
    safety_notice_fr: str = DERM_SAFETY_NOTICE_FR


class SameLesionLinkSchema(BaseModel):
    """Cross-modal SAME_LESION link between a text lesion entity and an ImageFinding."""

    link_id: str
    relation_type: Literal["SAME_LESION"] = "SAME_LESION"
    patient_id: str | None = None
    document_id: str
    image_id: str
    text_entity_id: str
    text_span: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    matched_body_site: str
    date_delta_days: int | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    review_status: ReviewStatusType = "needs_review"
    rationale_en: str = ""
    rationale_fr: str = ""


class RegionCoordinatesSchema(BaseModel):
    """Normalized page bounding box in [0, 1] coordinates."""

    x0: float = Field(ge=0.0, le=1.0)
    y0: float = Field(ge=0.0, le=1.0)
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)


class ExtractedRelationshipSchema(BaseModel):
    """Typed relationship between two extracted medical mentions."""

    relation_type: str
    target_value: str
    target_entity_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_text: str | None = None


class ExtractedRecordSchema(BaseModel):
    """Common structured output contract for all extraction approaches."""

    source_record_id: str
    document_id: str
    original_text: str
    extracted_value: str
    entity_type: str
    normalized_concept: str | None = None
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)
    page_number: int | None = Field(default=1, ge=1)
    region_coordinates: RegionCoordinatesSchema | None = None
    language: str
    negation_status: NegationStatusType
    assertion_status: AssertionStatusType
    certainty: CertaintyType = "confirmed"
    temporality: TemporalityType = "current"
    experiencer: ExperiencerType = "patient"
    uncertainty: float = Field(ge=0.0, le=1.0)
    temporal_information: str | None = None
    relationships: list[ExtractedRelationshipSchema] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)
    model_name: str
    model_version: str
    provenance: dict[str, Any] = Field(default_factory=dict)
    review_status: ReviewStatusType = "unreviewed"


class ExtractionResultSchema(BaseModel):
    """Top-level validated extraction output bundle."""

    source_record_id: str
    document_id: str
    language: str
    model_name: str
    model_version: str
    records: list[ExtractedRecordSchema] = Field(default_factory=list)
    image_findings: list[ImageFindingSchema] = Field(default_factory=list)
    same_lesion_links: list[SameLesionLinkSchema] = Field(default_factory=list)
    modality_status: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    clinical_notice: str = CLINICAL_RESEARCH_NOTICE


def export_extraction_json_schema(output_path: Path) -> Path:
    """Export the Pydantic v2 JSON Schema for ExtractionResultSchema to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema = ExtractionResultSchema.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    output_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    return output_path
