"""Canonical annotation schema for bilingual medical information extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

ENTITY_TYPES: tuple[str, ...] = (
    "SYMPTOM",
    "DIAGNOSIS",
    "MEDICATION",
    "DOSAGE",
    "FREQUENCY",
    "ROUTE",
    "DURATION",
    "LAB_TEST",
    "LAB_VALUE",
    "LAB_UNIT",
    "PROCEDURE",
    "ANATOMY",
    "DATE",
    "PATIENT_ID",
)

ASSERTION_CLASSES: tuple[str, ...] = (
    "present",
    "absent",
    "possible",
    "hypothetical",
    "family",
    "historical",
)

TEMPORALITY_CLASSES: tuple[str, ...] = ("current", "past", "planned")

RELATION_TYPES: tuple[str, ...] = (
    "TREATS",
    "HAS_DOSAGE",
    "HAS_FREQUENCY",
    "HAS_ROUTE",
    "HAS_VALUE",
    "HAS_UNIT",
    "REVEALS",
    "CAUSES",
    "LOCATED_IN",
)

TIMEX_TYPES: tuple[str, ...] = ("DATE", "DURATION", "FREQUENCY", "RELATIVE")

DOC_TYPES: tuple[str, ...] = (
    "discharge_summary",
    "consultation",
    "lab_report",
    "prescription",
    "radiology_report",
    "operative_note",
)


@dataclass(frozen=True)
class EntitySpan:
    """Character-offset-grounded entity annotation."""

    id: str
    type: str
    start: int
    end: int
    text: str
    assertion: str = "present"
    temporality: str = "current"
    normalized_code: str | None = None
    normalized_system: str | None = None


@dataclass(frozen=True)
class RelationSpan:
    """Directed relation between two EntitySpan IDs."""

    id: str
    type: str
    head: str
    tail: str


@dataclass(frozen=True)
class TimexSpan:
    """Temporal expression annotation anchored to document_date."""

    id: str
    type: str
    start: int
    end: int
    text: str
    normalized: str | None = None


@dataclass
class AnnotatedDocument:
    """Canonical document record with verified span offsets."""

    doc_id: str
    patient_id: str
    language: str
    doc_type: str
    document_date: str
    source: str
    text: str
    entities: list[EntitySpan] = field(default_factory=list)
    relations: list[RelationSpan] = field(default_factory=list)
    timex: list[TimexSpan] = field(default_factory=list)
    split: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate_spans(self) -> list[str]:
        """Verify that every entity and timex span matches text[start:end] and relations reference valid IDs."""
        errors: list[str] = []
        entity_ids = {e.id for e in self.entities}
        for e in self.entities:
            if e.type not in ENTITY_TYPES:
                errors.append(f"{self.doc_id}:{e.id} unknown entity type {e.type!r}")
            if e.assertion not in ASSERTION_CLASSES:
                errors.append(f"{self.doc_id}:{e.id} unknown assertion {e.assertion!r}")
            if e.temporality not in TEMPORALITY_CLASSES:
                errors.append(f"{self.doc_id}:{e.id} unknown temporality {e.temporality!r}")
            if e.start < 0 or e.end > len(self.text) or e.start >= e.end:
                errors.append(f"{self.doc_id}:{e.id} invalid offsets ({e.start}, {e.end})")
            elif self.text[e.start : e.end] != e.text:
                errors.append(
                    f"{self.doc_id}:{e.id} span mismatch: expected {e.text!r}, got {self.text[e.start : e.end]!r}"
                )
        for t in self.timex:
            if t.type not in TIMEX_TYPES:
                errors.append(f"{self.doc_id}:{t.id} unknown timex type {t.type!r}")
            if t.start < 0 or t.end > len(self.text) or t.start >= t.end:
                errors.append(f"{self.doc_id}:{t.id} invalid timex offsets ({t.start}, {t.end})")
            elif self.text[t.start : t.end] != t.text:
                errors.append(
                    f"{self.doc_id}:{t.id} timex mismatch: expected {t.text!r}, got {self.text[t.start : t.end]!r}"
                )
        for r in self.relations:
            if r.type not in RELATION_TYPES:
                errors.append(f"{self.doc_id}:{r.id} unknown relation type {r.type!r}")
            if r.head not in entity_ids or r.tail not in entity_ids:
                errors.append(f"{self.doc_id}:{r.id} dangling entity ref ({r.head} -> {r.tail})")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AnnotatedDocument:
        return cls(
            doc_id=str(payload["doc_id"]),
            patient_id=str(payload["patient_id"]),
            language=str(payload["language"]),
            doc_type=str(payload["doc_type"]),
            document_date=str(payload["document_date"]),
            source=str(payload.get("source", "unknown")),
            text=str(payload["text"]),
            entities=[EntitySpan(**e) for e in payload.get("entities", [])],
            relations=[RelationSpan(**r) for r in payload.get("relations", [])],
            timex=[TimexSpan(**t) for t in payload.get("timex", [])],
            split=payload.get("split"),
            metadata=dict(payload.get("metadata", {})),
        )
