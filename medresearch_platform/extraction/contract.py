"""Common output contract shared by all six medical extraction approaches."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

AssertionStatus = Literal["present", "absent", "uncertain", "hypothetical", "family_history", "historical"]
CertaintyStatus = Literal["confirmed", "possible", "absent"]
TemporalityStatus = Literal["current", "historical"]
ExperiencerStatus = Literal["patient", "family"]
NegationStatus = Literal["affirmed", "negated", "uncertain"]
ReviewStatus = Literal["unreviewed", "needs_review", "approved", "corrected", "rejected", "flagged", "image_unusable"]
RiskBand = Literal["low", "medium", "high", "abstained"]


@dataclass(frozen=True)
class ImageFinding:
    """Structured output contract for a dermatology/clinical image finding (IMAGE PROMPT Step 4)."""

    image_id: str
    document_id: str | None
    patient_id: str | None
    picture_type: str
    quality_result: dict[str, Any]
    malignancy_risk_probability: float | None
    malignancy_risk_band: RiskBand
    top3_diagnoses: tuple[dict[str, Any], ...]
    heatmap_uri: str | None
    abstained: bool
    abstain_reason_en: str | None
    abstain_reason_fr: str | None
    body_site: str | None
    model_name: str
    model_version: str
    git_commit: str
    review_status: ReviewStatus = "needs_review"
    provenance: dict[str, Any] = field(default_factory=dict)
    captured_date_iso: str | None = None
    safety_notice_en: str = "Research tool — not a diagnosis. A dermatologist must examine the lesion."
    safety_notice_fr: str = "Outil de recherche — ne constitue pas un diagnostic. Un dermatologue doit examiner la lésion."

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["top3_diagnoses"] = [dict(d) for d in self.top3_diagnoses]
        return payload


@dataclass(frozen=True)
class RegionCoordinates:
    """Normalized page bounding box in [0, 1] coordinates when available."""

    x0: float
    y0: float
    x1: float
    y1: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ExtractedRelationship:
    """Typed relationship between two extracted medical mentions."""

    relation_type: str
    target_value: str
    target_entity_type: str
    confidence: float
    evidence_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExtractedRecord:
    """Common structured output contract for all six extraction approaches (Spec §12)."""

    source_record_id: str
    document_id: str
    original_text: str
    extracted_value: str
    entity_type: str
    normalized_concept: str | None
    char_start: int | None
    char_end: int | None
    page_number: int | None
    region_coordinates: RegionCoordinates | None
    language: str
    negation_status: NegationStatus
    assertion_status: AssertionStatus
    uncertainty: float
    temporal_information: str | None
    relationships: tuple[ExtractedRelationship, ...]
    confidence_score: float
    model_name: str
    model_version: str
    provenance: dict[str, Any]
    review_status: ReviewStatus = "unreviewed"
    confidence_variance: float = 0.0
    high_uncertainty: bool = False
    certainty: CertaintyStatus = "confirmed"
    temporality: TemporalityStatus = "current"
    experiencer: ExperiencerStatus = "patient"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["relationships"] = [rel.to_dict() for rel in self.relationships]
        if self.region_coordinates is not None:
            payload["region_coordinates"] = self.region_coordinates.to_dict()
        return payload


@dataclass(frozen=True)
class ExtractionResult:
    """Container for all records produced by one extraction approach on a document."""

    source_record_id: str
    document_id: str
    language: str
    model_name: str
    model_version: str
    records: tuple[ExtractedRecord, ...] = field(default_factory=tuple)
    image_findings: tuple[ImageFinding, ...] = field(default_factory=tuple)
    same_lesion_links: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    modality_status: dict[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_record_id": self.source_record_id,
            "document_id": self.document_id,
            "language": self.language,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "records": [record.to_dict() for record in self.records],
            "image_findings": [f.to_dict() for f in self.image_findings],
            "same_lesion_links": [dict(link) for link in self.same_lesion_links],
            "modality_status": dict(self.modality_status),
            "warnings": list(self.warnings),
        }


def build_extracted_record(
    *,
    source_record_id: str,
    document_id: str,
    original_text: str,
    extracted_value: str,
    entity_type: str,
    normalized_concept: str | None,
    char_start: int | None,
    char_end: int | None,
    language: str,
    negation_status: NegationStatus,
    assertion_status: AssertionStatus,
    temporal_information: str | None,
    confidence_score: float,
    model_name: str,
    model_version: str,
    provenance: dict[str, Any],
    page_number: int | None = 1,
    region_coordinates: RegionCoordinates | None = None,
    relationships: tuple[ExtractedRelationship, ...] = (),
    review_status: ReviewStatus = "unreviewed",
    certainty: CertaintyStatus | None = None,
    temporality: TemporalityStatus | None = None,
    experiencer: ExperiencerStatus | None = None,
) -> ExtractedRecord:
    """Construct an ExtractedRecord with consistent uncertainty and 3-axis assertion attributes."""
    bounded_conf = max(0.0, min(1.0, float(confidence_score)))
    uncertainty = round(1.0 - bounded_conf, 4)
    if assertion_status == "uncertain":
        uncertainty = max(uncertainty, 0.35)
    eff_certainty: CertaintyStatus = certainty or (
        "absent"
        if (assertion_status == "absent" or negation_status == "negated")
        else ("possible" if assertion_status in {"uncertain", "hypothetical"} else "confirmed")
    )
    eff_temporality: TemporalityStatus = temporality or (
        "historical" if assertion_status == "historical" else "current"
    )
    eff_experiencer: ExperiencerStatus = experiencer or (
        "family" if assertion_status == "family_history" else "patient"
    )
    return ExtractedRecord(
        source_record_id=source_record_id,
        document_id=document_id,
        original_text=original_text,
        extracted_value=extracted_value,
        entity_type=entity_type,
        normalized_concept=normalized_concept,
        char_start=char_start,
        char_end=char_end,
        page_number=page_number,
        region_coordinates=region_coordinates,
        language=language,
        negation_status=negation_status,
        assertion_status=assertion_status,
        uncertainty=round(uncertainty, 4),
        temporal_information=temporal_information,
        relationships=relationships,
        confidence_score=round(bounded_conf, 4),
        model_name=model_name,
        model_version=model_version,
        provenance=dict(provenance),
        review_status=review_status,
        certainty=eff_certainty,
        temporality=eff_temporality,
        experiencer=eff_experiencer,
    )


def map_assertion_and_negation(
    raw_assertion: str,
    raw_negation: str,
    temporal_status: str,
) -> tuple[AssertionStatus, NegationStatus]:
    """Normalize cue-level assertion/negation strings to the common contract literals."""
    assertion_map: dict[str, AssertionStatus] = {
        "AFFIRMED": "present",
        "NEGATED": "absent",
        "UNCERTAIN": "uncertain",
        "HYPOTHETICAL": "hypothetical",
        "FAMILY_HISTORY": "family_history",
    }
    assertion: AssertionStatus = assertion_map.get(raw_assertion, "present")
    if assertion == "present" and temporal_status == "HISTORICAL":
        assertion = "historical"

    negation_map: dict[str, NegationStatus] = {
        "AFFIRMED": "affirmed",
        "NEGATED": "negated",
        "UNKNOWN": "uncertain",
    }
    negation: NegationStatus = negation_map.get(raw_negation, "uncertain")
    return assertion, negation


def link_adjacent_relationships(
    spans: list[tuple[int, int, str, str, float]],
    text: str,
) -> dict[int, tuple[ExtractedRelationship, ...]]:
    """Build conservative proximity relationships (e.g. MEDICATION->DOSAGE/FREQUENCY/ROUTE, LAB_TEST->LAB_VALUE/UNIT)."""
    rels: dict[int, list[ExtractedRelationship]] = {i: [] for i in range(len(spans))}
    for i, (s1, e1, t1, v1, c1) in enumerate(spans):
        for j, (s2, e2, t2, v2, c2) in enumerate(spans):
            if i == j:
                continue
            gap = abs(s2 - e1) if s2 >= e1 else abs(s1 - e2)
            if gap > 60:
                continue
            rel_type: str | None = None
            if t1 == "MEDICATION" and t2 in {"DOSAGE", "FREQUENCY", "ROUTE"}:
                rel_type = f"HAS_{t2}"
            elif t1 == "LAB_TEST" and t2 in {"LAB_VALUE", "LAB_UNIT"}:
                rel_type = f"HAS_{t2}"
            if rel_type is not None:
                lo, hi = min(s1, s2), max(e1, e2)
                rels[i].append(
                    ExtractedRelationship(
                        relation_type=rel_type,
                        target_value=v2,
                        target_entity_type=t2,
                        confidence=round(min(c1, c2), 4),
                        evidence_text=text[lo:hi],
                    )
                )
    return {i: tuple(items) for i, items in rels.items()}


def records_from_bio_tags(
    *,
    text: str,
    tokens: list[Any],
    tags: list[str],
    token_confidences: list[float],
    source_record_id: str,
    document_id: str,
    language: str,
    model_name: str,
    model_version: str,
    provenance: dict[str, Any],
    terminology: Any | None = None,
    page_number: int = 1,
) -> ExtractionResult:
    """Convert BIO tag predictions + token confidences into an ExtractionResult."""
    from medresearch_platform.ai.terminology import BilingualTerminologyRegistry
    from medresearch_platform.extraction.text import assess_assertion, bio_to_spans

    registry = terminology or BilingualTerminologyRegistry()
    raw_spans = bio_to_spans(tokens, tags)
    span_items: list[tuple[int, int, str, str, float]] = []
    for char_start, char_end, entity_type, tok_start, tok_end in raw_spans:
        value = text[char_start:char_end]
        confs = token_confidences[tok_start:tok_end]
        conf = float(sum(confs) / len(confs)) if confs else 0.5
        span_items.append((char_start, char_end, entity_type, value, conf))

    rels = link_adjacent_relationships(span_items, text)
    records: list[ExtractedRecord] = []
    for idx, (char_start, char_end, entity_type, value, conf) in enumerate(span_items):
        raw_assert, raw_neg, temporal = assess_assertion(text, char_start, char_end)
        assertion, negation = map_assertion_and_negation(raw_assert, raw_neg, temporal)
        mapping = registry.map_mention(value, category_hint=entity_type)
        records.append(
            build_extracted_record(
                source_record_id=source_record_id,
                document_id=document_id,
                original_text=value,
                extracted_value=value,
                entity_type=entity_type,
                normalized_concept=mapping.concept_id,
                char_start=char_start,
                char_end=char_end,
                language=language,
                negation_status=negation,
                assertion_status=assertion,
                temporal_information=temporal,
                confidence_score=conf,
                model_name=model_name,
                model_version=model_version,
                provenance={
                    **provenance,
                    "mapping_match_type": mapping.match_type,
                },
                page_number=page_number,
                relationships=rels.get(idx, ()),
            )
        )

    return ExtractionResult(
        source_record_id=source_record_id,
        document_id=document_id,
        language=language,
        model_name=model_name,
        model_version=model_version,
        records=tuple(records),
        modality_status={"text": "available"},
    )


