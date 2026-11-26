"""Versioned Bilingual (English & French) Terminology Registry and Mapper (Section 14)."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Literal

MatchType = Literal["EXACT", "SYNONYM", "APPROXIMATE", "AMBIGUOUS", "UNMAPPED"]


@dataclass(frozen=True)
class TerminologyConcept:
    concept_id: str
    vocabulary: str
    code: str
    preferred_term_en: str
    preferred_term_fr: str
    synonyms_en: tuple[str, ...]
    synonyms_fr: tuple[str, ...]
    concept_category: str
    version: str = "2026.1-DEMO"


@dataclass(frozen=True)
class TerminologyMappingResult:
    source_text: str
    normalized_text: str
    concept_id: str | None
    vocabulary: str | None
    code: str | None
    match_type: MatchType
    confidence: float
    candidate_codes: tuple[str, ...] = ()


DEFAULT_BILINGUAL_CONCEPTS: tuple[TerminologyConcept, ...] = (
    # Diagnoses
    TerminologyConcept(
        concept_id="term_hypertension",
        vocabulary="ICD-10",
        code="I10",
        preferred_term_en="essential hypertension",
        preferred_term_fr="hypertension artérielle essentielle",
        synonyms_en=("hypertension", "high blood pressure", "htn", "arterial hypertension"),
        synonyms_fr=("hypertension artérielle", "hta", "hypertension"),
        concept_category="DIAGNOSIS",
    ),
    TerminologyConcept(
        concept_id="term_t2dm",
        vocabulary="ICD-10",
        code="E11.9",
        preferred_term_en="type 2 diabetes mellitus",
        preferred_term_fr="diabète sucré de type 2",
        synonyms_en=("type 2 diabetes", "t2dm", "diabetes mellitus type 2", "non-insulin-dependent diabetes"),
        synonyms_fr=("diabète de type 2", "dt2", "diabète type 2"),
        concept_category="DIAGNOSIS",
    ),
    TerminologyConcept(
        concept_id="term_pneumonia",
        vocabulary="ICD-10",
        code="J18.9",
        preferred_term_en="pneumonia",
        preferred_term_fr="pneumopathie infectieuse",
        synonyms_en=("bacterial pneumonia", "pulmonary infection", "community-acquired pneumonia"),
        synonyms_fr=("pneumonie", "pneumopathie", "infection pulmonaire"),
        concept_category="DIAGNOSIS",
    ),
    TerminologyConcept(
        concept_id="term_pe",
        vocabulary="ICD-10",
        code="I26.9",
        preferred_term_en="pulmonary embolism",
        preferred_term_fr="embolie pulmonaire",
        synonyms_en=("pe", "thromboembolism pulmonary"),
        synonyms_fr=("ep", "thrombo-embolie pulmonaire"),
        concept_category="DIAGNOSIS",
    ),
    TerminologyConcept(
        concept_id="term_afib",
        vocabulary="ICD-10",
        code="I48.91",
        preferred_term_en="atrial fibrillation",
        preferred_term_fr="fibrillation auriculaire",
        synonyms_en=("afib", "a-fib", "auricular fibrillation"),
        synonyms_fr=("fa", "fibrillation atriale", "acfa"),
        concept_category="DIAGNOSIS",
    ),
    # Symptoms
    TerminologyConcept(
        concept_id="term_dyspnea",
        vocabulary="SNOMED-CT",
        code="267036007",
        preferred_term_en="dyspnea",
        preferred_term_fr="dyspnée",
        synonyms_en=("shortness of breath", "sob", "breathlessness", "respiratory distress"),
        synonyms_fr=("essoufflement", "gêne respiratoire", "dyspnee"),
        concept_category="SYMPTOM",
    ),
    TerminologyConcept(
        concept_id="term_chest_pain",
        vocabulary="SNOMED-CT",
        code="29857009",
        preferred_term_en="chest pain",
        preferred_term_fr="douleur thoracique",
        synonyms_en=("thoracic pain", "retrosternal pain", "precordial pain"),
        synonyms_fr=("douleurs thoraciques", "douleur rétro-sternale"),
        concept_category="SYMPTOM",
    ),
    TerminologyConcept(
        concept_id="term_fever",
        vocabulary="SNOMED-CT",
        code="386661006",
        preferred_term_en="fever",
        preferred_term_fr="fièvre",
        synonyms_en=("pyrexia", "febrile state", "hyperthermia"),
        synonyms_fr=("fievre", "état fébrile", "hyperthermie"),
        concept_category="SYMPTOM",
    ),
    TerminologyConcept(
        concept_id="term_cough",
        vocabulary="SNOMED-CT",
        code="49727002",
        preferred_term_en="cough",
        preferred_term_fr="toux",
        synonyms_en=("productive cough", "dry cough"),
        synonyms_fr=("toux sèche", "toux grasse"),
        concept_category="SYMPTOM",
    ),
    # Medications
    TerminologyConcept(
        concept_id="term_metformin",
        vocabulary="RXNORM",
        code="6809",
        preferred_term_en="metformin",
        preferred_term_fr="metformine",
        synonyms_en=("glucophage", "metformin hydrochloride"),
        synonyms_fr=("glucophage", "chlorhydrate de metformine"),
        concept_category="MEDICATION",
    ),
    TerminologyConcept(
        concept_id="term_amlodipine",
        vocabulary="RXNORM",
        code="17767",
        preferred_term_en="amlodipine",
        preferred_term_fr="amlodipine",
        synonyms_en=("norvasc", "amlodipine besylate"),
        synonyms_fr=("amlor", "bésilate d'amlodipine"),
        concept_category="MEDICATION",
    ),
    TerminologyConcept(
        concept_id="term_amoxicillin",
        vocabulary="ATC",
        code="J01CA04",
        preferred_term_en="amoxicillin",
        preferred_term_fr="amoxicilline",
        synonyms_en=("amoxil", "amoxicillin trihydrate"),
        synonyms_fr=("clamoxyl", "amoxicilline"),
        concept_category="MEDICATION",
    ),
    TerminologyConcept(
        concept_id="term_apixaban",
        vocabulary="RXNORM",
        code="1364430",
        preferred_term_en="apixaban",
        preferred_term_fr="apixaban",
        synonyms_en=("eliquis",),
        synonyms_fr=("eliquis",),
        concept_category="MEDICATION",
    ),
    # Laboratory Tests
    TerminologyConcept(
        concept_id="term_hba1c",
        vocabulary="LOINC",
        code="4548-4",
        preferred_term_en="hemoglobin a1c",
        preferred_term_fr="hémoglobine glyquée",
        synonyms_en=("hba1c", "glycated hemoglobin", "a1c"),
        synonyms_fr=("hba1c", "hemoglobine glyquee"),
        concept_category="LAB_TEST",
    ),
    TerminologyConcept(
        concept_id="term_crp",
        vocabulary="LOINC",
        code="1988-5",
        preferred_term_en="c-reactive protein",
        preferred_term_fr="protéine c-réactive",
        synonyms_en=("crp", "c reactive protein"),
        synonyms_fr=("crp", "proteine c reactive"),
        concept_category="LAB_TEST",
    ),
    TerminologyConcept(
        concept_id="term_creatinine",
        vocabulary="LOINC",
        code="2160-0",
        preferred_term_en="serum creatinine",
        preferred_term_fr="créatininémie",
        synonyms_en=("creatinine", "blood creatinine"),
        synonyms_fr=("créatinine", "creatinine", "creatininemie"),
        concept_category="LAB_TEST",
    ),
    # Anatomy & Procedures
    TerminologyConcept(
        concept_id="term_thorax",
        vocabulary="SNOMED-CT",
        code="51185008",
        preferred_term_en="thorax",
        preferred_term_fr="thorax",
        synonyms_en=("chest", "thoracic cavity", "lungs", "pulmonary"),
        synonyms_fr=("poitrine", "poumons", "pulmonaire", "thoracique"),
        concept_category="ANATOMY",
    ),
    TerminologyConcept(
        concept_id="term_chest_ct",
        vocabulary="SNOMED-CT",
        code="169069000",
        preferred_term_en="computed tomography of chest",
        preferred_term_fr="angioscanner thoracique",
        synonyms_en=("chest ct", "ct pulmonary angiogram", "ctpa", "thoracic ct"),
        synonyms_fr=("scanner thoracique", "tdm thoracique"),
        concept_category="PROCEDURE",
    ),
)


class BilingualTerminologyRegistry:
    """In-memory and relational terminology lookup preserving ambiguity and unmapped terms."""

    def __init__(self, concepts: tuple[TerminologyConcept, ...] = DEFAULT_BILINGUAL_CONCEPTS) -> None:
        self.concepts = concepts

    def sync_to_database(self, conn: sqlite3.Connection) -> int:
        count = 0
        for c in self.concepts:
            conn.execute(
                """
                INSERT OR REPLACE INTO terminology_concepts (
                    concept_id, vocabulary, code, preferred_term_en, preferred_term_fr,
                    synonyms_en_json, synonyms_fr_json, concept_category, version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    c.concept_id,
                    c.vocabulary,
                    c.code,
                    c.preferred_term_en,
                    c.preferred_term_fr,
                    json.dumps(list(c.synonyms_en)),
                    json.dumps(list(c.synonyms_fr)),
                    c.concept_category,
                    c.version,
                ),
            )
            count += 1
        return count

    def map_mention(self, surface_text: str, category_hint: str | None = None) -> TerminologyMappingResult:
        """Map an English or French clinical mention to controlled terminology without forcing uncertain matches."""
        cleaned = " ".join(surface_text.strip().lower().split())
        if not cleaned:
            return TerminologyMappingResult(
                source_text=surface_text,
                normalized_text="",
                concept_id=None,
                vocabulary=None,
                code=None,
                match_type="UNMAPPED",
                confidence=0.0,
            )

        exact_hits: list[TerminologyConcept] = []
        synonym_hits: list[TerminologyConcept] = []
        approx_hits: list[TerminologyConcept] = []

        for concept in self.concepts:
            if category_hint and concept.concept_category != category_hint:
                continue
            if cleaned in (concept.preferred_term_en.lower(), concept.preferred_term_fr.lower()):
                exact_hits.append(concept)
                continue
            syns = {s.lower() for s in concept.synonyms_en} | {s.lower() for s in concept.synonyms_fr}
            if cleaned in syns:
                synonym_hits.append(concept)
                continue
            all_terms = [concept.preferred_term_en.lower(), concept.preferred_term_fr.lower(), *syns]
            if any(len(cleaned) >= 4 and (cleaned in t or t in cleaned) for t in all_terms):
                approx_hits.append(concept)

        if len(exact_hits) == 1:
            hit = exact_hits[0]
            return TerminologyMappingResult(
                source_text=surface_text,
                normalized_text=hit.preferred_term_en,
                concept_id=hit.concept_id,
                vocabulary=hit.vocabulary,
                code=hit.code,
                match_type="EXACT",
                confidence=0.99,
                candidate_codes=(f"{hit.vocabulary}:{hit.code}",),
            )
        if len(synonym_hits) == 1:
            hit = synonym_hits[0]
            return TerminologyMappingResult(
                source_text=surface_text,
                normalized_text=hit.preferred_term_en,
                concept_id=hit.concept_id,
                vocabulary=hit.vocabulary,
                code=hit.code,
                match_type="SYNONYM",
                confidence=0.94,
                candidate_codes=(f"{hit.vocabulary}:{hit.code}",),
            )

        combined = exact_hits + synonym_hits + approx_hits
        if len(combined) > 1:
            return TerminologyMappingResult(
                source_text=surface_text,
                normalized_text=cleaned,
                concept_id=None,
                vocabulary=None,
                code=None,
                match_type="AMBIGUOUS",
                confidence=0.45,
                candidate_codes=tuple(f"{c.vocabulary}:{c.code}" for c in combined),
            )
        if len(approx_hits) == 1:
            hit = approx_hits[0]
            return TerminologyMappingResult(
                source_text=surface_text,
                normalized_text=hit.preferred_term_en,
                concept_id=hit.concept_id,
                vocabulary=hit.vocabulary,
                code=hit.code,
                match_type="APPROXIMATE",
                confidence=0.78,
                candidate_codes=(f"{hit.vocabulary}:{hit.code}",),
            )

        return TerminologyMappingResult(
            source_text=surface_text,
            normalized_text=cleaned,
            concept_id=None,
            vocabulary=None,
            code=None,
            match_type="UNMAPPED",
            confidence=0.0,
        )
