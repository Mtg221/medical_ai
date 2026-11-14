"""Phase 1 Data Foundation: schema, converters, synthetic bilingual generator v2, splits, and builder."""

from medresearch_platform.data.schema import (
    ASSERTION_CLASSES,
    DOC_TYPES,
    ENTITY_TYPES,
    RELATION_TYPES,
    TEMPORALITY_CLASSES,
    TIMEX_TYPES,
    AnnotatedDocument,
    EntitySpan,
    RelationSpan,
    TimexSpan,
)

__all__ = [
    "ASSERTION_CLASSES",
    "AnnotatedDocument",
    "DOC_TYPES",
    "ENTITY_TYPES",
    "EntitySpan",
    "RELATION_TYPES",
    "RelationSpan",
    "TEMPORALITY_CLASSES",
    "TIMEX_TYPES",
    "TimexSpan",
]
