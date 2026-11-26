"""Approach 1: Versioned, explainable bilingual rule-based medical extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass

from medresearch_platform.ai.nlp_model import _BILINGUAL_SPAN_PATTERNS
from medresearch_platform.ai.terminology import BilingualTerminologyRegistry
from medresearch_platform.extraction.contract import (
    ExtractedRecord,
    ExtractionResult,
    build_extracted_record,
    link_adjacent_relationships,
    map_assertion_and_negation,
)
from medresearch_platform.extraction.text import ENTITY_TYPES, assess_assertion


@dataclass(frozen=True)
class RuleMatch:
    rule_id: str
    entity_type: str
    char_start: int
    char_end: int
    matched_text: str
    context_window: str
    confidence: float


class RuleBasedMedicalExtractor:
    """Deterministic, explainable bilingual rule-based medical entity & context extractor."""

    model_name = "RuleBasedMedicalExtractor"
    model_version = "1.0.0"

    def __init__(self, terminology: BilingualTerminologyRegistry | None = None) -> None:
        self.terminology = terminology or BilingualTerminologyRegistry()
        self.patterns: tuple[tuple[str, str, re.Pattern[str]], ...] = tuple(
            (f"rule_{etype.lower()}_v1", etype, pattern)
            for etype, pattern in _BILINGUAL_SPAN_PATTERNS
            if etype in ENTITY_TYPES
        )

    def extract(
        self,
        text: str,
        *,
        source_record_id: str = "record-0",
        document_id: str = "doc-0",
        language: str = "en",
        page_number: int = 1,
    ) -> ExtractionResult:
        if not text.strip():
            return ExtractionResult(
                source_record_id=source_record_id,
                document_id=document_id,
                language=language,
                model_name=self.model_name,
                model_version=self.model_version,
                records=(),
                modality_status={"text": "absent"},
                warnings=("empty_text_input",),
            )

        candidates: list[RuleMatch] = []
        for rule_id, entity_type, pattern in self.patterns:
            for match in pattern.finditer(text):
                start, end = (match.start(1), match.end(1)) if match.lastindex else (match.start(), match.end())
                lo = max(0, start - 35)
                hi = min(len(text), end + 35)
                candidates.append(
                    RuleMatch(
                        rule_id=rule_id,
                        entity_type=entity_type,
                        char_start=start,
                        char_end=end,
                        matched_text=text[start:end],
                        context_window=text[lo:hi],
                        confidence=0.92,
                    )
                )

        # Resolve overlapping matches by keeping longer spans first, then earlier start.
        candidates.sort(key=lambda m: (-(m.char_end - m.char_start), m.char_start))
        selected: list[RuleMatch] = []
        for cand in candidates:
            if any(not (cand.char_end <= s.char_start or cand.char_start >= s.char_end) for s in selected):
                continue
            selected.append(cand)
        selected.sort(key=lambda m: m.char_start)

        span_items = [
            (m.char_start, m.char_end, m.entity_type, m.matched_text, m.confidence)
            for m in selected
        ]
        rels = link_adjacent_relationships(span_items, text)

        records: list[ExtractedRecord] = []
        for idx, m in enumerate(selected):
            raw_assert, raw_neg, temporal = assess_assertion(text, m.char_start, m.char_end)
            assertion, negation = map_assertion_and_negation(raw_assert, raw_neg, temporal)
            mapping = self.terminology.map_mention(m.matched_text, category_hint=m.entity_type)
            records.append(
                build_extracted_record(
                    source_record_id=source_record_id,
                    document_id=document_id,
                    original_text=m.matched_text,
                    extracted_value=m.matched_text,
                    entity_type=m.entity_type,
                    normalized_concept=mapping.concept_id,
                    char_start=m.char_start,
                    char_end=m.char_end,
                    language=language,
                    negation_status=negation,
                    assertion_status=assertion,
                    temporal_information=temporal,
                    confidence_score=m.confidence,
                    model_name=self.model_name,
                    model_version=self.model_version,
                    provenance={
                        "approach": "rule_based",
                        "rule_id": m.rule_id,
                        "context_window": m.context_window,
                        "mapping_match_type": mapping.match_type,
                        "pretrained_weights_used": False,
                    },
                    page_number=page_number,
                    relationships=rels.get(idx, ()),
                )
            )

        return ExtractionResult(
            source_record_id=source_record_id,
            document_id=document_id,
            language=language,
            model_name=self.model_name,
            model_version=self.model_version,
            records=tuple(records),
            modality_status={"text": "available"},
        )
