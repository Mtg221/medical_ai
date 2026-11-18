"""SYNTH-MED-V3 bilingual clinical corpus generator with generalization splits (F2 fix)."""

from __future__ import annotations

import random
import re
from datetime import date, timedelta
from typing import Any

from medresearch_platform.data.lexicon_v3 import (
    ANATOMY_EN,
    ANATOMY_FR,
    DIAGNOSES_EN,
    DIAGNOSES_FR,
    LABS_EN,
    LABS_FR,
    MEDICATIONS_EN,
    MEDICATIONS_FR,
    PROCEDURES_EN,
    PROCEDURES_FR,
    SYMPTOMS_EN,
    SYMPTOMS_FR,
    LabEntry,
    LexiconEntry,
    MedEntry,
    partition_concepts,
)
from medresearch_platform.data.schema import DOC_TYPES, AnnotatedDocument, EntitySpan
from medresearch_platform.data.synthetic_v2 import _DocBuilder
from medresearch_platform.data.templates_v3 import (
    TemplateSpec,
    build_all_templates,
    partition_templates,
)

_SLOT_RE = re.compile(r"\{([A-Z0-9_]+)\}")

_MONTHS_EN = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
_MONTHS_FR = (
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
)


def _ordinal_en(day: int) -> str:
    if 11 <= (day % 100) <= 13:
        return f"{day}th"
    return f"{day}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th') }"


def _format_date_with_norm(d: date, lang: str, rng: random.Random) -> tuple[str, str]:
    style = rng.randrange(5)
    if style == 0:
        return d.isoformat(), d.isoformat()
    if style == 1:
        return f"{d.day:02d}/{d.month:02d}/{d.year}", d.isoformat()
    if style == 2:
        if lang == "en":
            return f"{_MONTHS_EN[d.month - 1]} {d.day}, {d.year}", d.isoformat()
        return f"{d.day} {_MONTHS_FR[d.month - 1]} {d.year}", d.isoformat()
    if style == 3:
        if lang == "en":
            return f"{_MONTHS_EN[d.month - 1]} {_ordinal_en(d.day)}", d.isoformat()
        return f"{d.day} {_MONTHS_FR[d.month - 1]}", d.isoformat()
    # Partial month+year
    if lang == "en":
        return f"{_MONTHS_EN[d.month - 1]} {d.year}", f"{d.year}-{d.month:02d}"
    return f"{_MONTHS_FR[d.month - 1]} {d.year}", f"{d.year}-{d.month:02d}"


def _sample_rel_timex(doc_date: date, lang: str, rng: random.Random) -> tuple[str, str]:
    if lang == "en":
        choices = [
            ("yesterday", (doc_date - timedelta(days=1)).isoformat()),
            ("3 days ago", (doc_date - timedelta(days=3)).isoformat()),
            ("2 weeks ago", (doc_date - timedelta(weeks=2)).isoformat()),
            ("1 month ago", (doc_date - timedelta(days=30)).isoformat()),
            ("POD 3", doc_date.isoformat()),
            ("in 2 weeks", (doc_date + timedelta(weeks=2)).isoformat()),
        ]
    else:
        choices = [
            ("la veille", (doc_date - timedelta(days=1)).isoformat()),
            ("il y a 3 jours", (doc_date - timedelta(days=3)).isoformat()),
            ("depuis 3 semaines", (doc_date - timedelta(weeks=3)).isoformat()),
            ("il y a 1 mois", (doc_date - timedelta(days=30)).isoformat()),
            ("J3 post-op", doc_date.isoformat()),
            ("dans 2 semaines", (doc_date + timedelta(weeks=2)).isoformat()),
        ]
    return rng.choice(choices)


def _sample_duration(lang: str, rng: random.Random) -> tuple[str, str]:
    if lang == "en":
        choices = [
            ("for 5 days", "P5D"),
            ("for 7 days", "P7D"),
            ("for 10 days", "P10D"),
            ("for 2 weeks", "P2W"),
            ("for 3 months", "P3M"),
        ]
    else:
        choices = [
            ("pendant 5 jours", "P5D"),
            ("pendant 7 jours", "P7D"),
            ("durant 10 jours", "P10D"),
            ("pendant 2 semaines", "P2W"),
            ("pendant 3 mois", "P3M"),
        ]
    return rng.choice(choices)


def _sample_frequency(lang: str, rng: random.Random) -> tuple[str, str]:
    if lang == "en":
        choices = [
            ("once daily", "QD"),
            ("twice daily", "BID"),
            ("b.i.d.", "BID"),
            ("three times daily", "TID"),
            ("t.i.d.", "TID"),
            ("as needed", "PRN"),
        ]
    else:
        choices = [
            ("1 fois par jour", "QD"),
            ("matin et soir", "BID"),
            ("1 cp x 2/j", "BID"),
            ("3 fois par jour", "TID"),
            ("toutes les 8 heures", "TID"),
            ("si besoin", "PRN"),
        ]
    return rng.choice(choices)


def _surface_variant(text: str, rng: random.Random, *, allow_case: bool = True) -> str:
    if allow_case and len(text) > 5 and not text.isupper() and rng.random() < 0.15:
        return text[0].upper() + text[1:] if text[0].islower() else text[0].lower() + text[1:]
    return text


class _LexiconPool:
    """Holds language-specific lexicon pools (either seen 70% or unseen 30%)."""

    def __init__(self, lang: str, *, use_unseen: bool, seed: int) -> None:
        self.lang = lang
        dx_all = DIAGNOSES_EN if lang == "en" else DIAGNOSES_FR
        sx_all = SYMPTOMS_EN if lang == "en" else SYMPTOMS_FR
        med_all = MEDICATIONS_EN if lang == "en" else MEDICATIONS_FR
        lab_all = LABS_EN if lang == "en" else LABS_FR
        proc_all = PROCEDURES_EN if lang == "en" else PROCEDURES_FR
        anat_all = ANATOMY_EN if lang == "en" else ANATOMY_FR

        idx = 1 if use_unseen else 0
        self.diagnoses: list[LexiconEntry] = partition_concepts(dx_all, seed=seed)[idx]
        self.symptoms: list[LexiconEntry] = partition_concepts(sx_all, seed=seed + 1)[idx]
        self.medications: list[MedEntry] = partition_concepts(med_all, seed=seed + 2)[idx]
        self.labs: list[LabEntry] = partition_concepts(lab_all, seed=seed + 3)[idx]
        self.procedures: list[LexiconEntry] = partition_concepts(proc_all, seed=seed + 4)[idx]
        self.anatomy: list[LexiconEntry] = partition_concepts(anat_all, seed=seed + 5)[idx]


def _render_template(
    b: _DocBuilder,
    pattern: str,
    pool: _LexiconPool,
    lang: str,
    patient_id: str,
    doc_date: date,
    rng: random.Random,
) -> None:
    """Parse `pattern` slots, emit text/entities/TIMEX/relations with exact offsets."""
    # Pre-sample paired objects so relations within the same sentence link cleanly
    med1 = rng.choice(pool.medications)
    med2 = rng.choice(pool.medications)
    lab1 = rng.choice(pool.labs)
    lab2 = rng.choice(pool.labs)
    dx_pos = rng.choice(pool.diagnoses)
    proc1 = rng.choice(pool.procedures)
    anat1 = rng.choice(pool.anatomy)

    med1_ent: EntitySpan | None = None
    med2_ent: EntitySpan | None = None
    lab1_ent: EntitySpan | None = None
    lab2_ent: EntitySpan | None = None
    dx_pos_ent: EntitySpan | None = None

    pos = 0
    for m in _SLOT_RE.finditer(pattern):
        if m.start() > pos:
            b.add_text(pattern[pos : m.start()])
        slot = m.group(1)

        if slot == "PAT":
            b.add_entity(patient_id, etype="PATIENT_ID", assertion="present")
        elif slot == "DATE":
            txt, norm = _format_date_with_norm(doc_date, lang, rng)
            b.add_entity_with_timex(txt, etype="DATE", ttype="DATE", normalized=norm)
        elif slot == "REL_TIMEX":
            txt, norm = _sample_rel_timex(doc_date, lang, rng)
            b.add_entity_with_timex(txt, etype="DATE", ttype="RELATIVE", normalized=norm)
        elif slot == "DUR":
            txt, norm = _sample_duration(lang, rng)
            b.add_entity_with_timex(txt, etype="DURATION", ttype="DURATION", normalized=norm)
        elif slot == "FREQ":
            txt, norm = _sample_frequency(lang, rng)
            ent, _ = b.add_entity_with_timex(txt, etype="FREQUENCY", ttype="FREQUENCY", normalized=norm)
            if med1_ent is not None:
                b.add_relation("HAS_FREQUENCY", med1_ent, ent)
        elif slot == "FREQ2":
            txt, norm = _sample_frequency(lang, rng)
            ent, _ = b.add_entity_with_timex(txt, etype="FREQUENCY", ttype="FREQUENCY", normalized=norm)
            if med2_ent is not None:
                b.add_relation("HAS_FREQUENCY", med2_ent, ent)
        elif slot in {"SX_POS", "SX_POS2"}:
            sx = rng.choice(pool.symptoms)
            b.add_entity(_surface_variant(rng.choice(sx.variants), rng), etype="SYMPTOM", assertion="present", code=sx.code, system=sx.system)
        elif slot in {"SX_NEG", "SX_NEG2"}:
            sx = rng.choice(pool.symptoms)
            b.add_entity(_surface_variant(rng.choice(sx.variants), rng), etype="SYMPTOM", assertion="absent", code=sx.code, system=sx.system)
        elif slot == "SX_HYP":
            sx = rng.choice(pool.symptoms)
            b.add_entity(_surface_variant(rng.choice(sx.variants), rng), etype="SYMPTOM", assertion="hypothetical", code=sx.code, system=sx.system)
        elif slot == "DX_POS":
            dx_pos_ent = b.add_entity(_surface_variant(rng.choice(dx_pos.variants), rng), etype="DIAGNOSIS", assertion="present", code=dx_pos.code, system=dx_pos.system)
        elif slot == "DX_NEG":
            dx = rng.choice(pool.diagnoses)
            b.add_entity(_surface_variant(rng.choice(dx.variants), rng), etype="DIAGNOSIS", assertion="absent", code=dx.code, system=dx.system)
        elif slot == "DX_UNC":
            dx = rng.choice(pool.diagnoses)
            b.add_entity(_surface_variant(rng.choice(dx.variants), rng), etype="DIAGNOSIS", assertion="possible", code=dx.code, system=dx.system)
        elif slot in {"DX_HIST", "DX_HIST2"}:
            dx = rng.choice(pool.diagnoses)
            b.add_entity(_surface_variant(rng.choice(dx.variants), rng), etype="DIAGNOSIS", assertion="historical", temporality="past", code=dx.code, system=dx.system)
        elif slot == "DX_FAM":
            dx = rng.choice(pool.diagnoses)
            b.add_entity(_surface_variant(rng.choice(dx.variants), rng), etype="DIAGNOSIS", assertion="family", temporality="past", code=dx.code, system=dx.system)
        elif slot == "MED":
            med1_ent = b.add_entity(rng.choice(med1.variants), etype="MEDICATION", assertion="present", code=med1.code, system=med1.system)
        elif slot == "MED2":
            med2_ent = b.add_entity(rng.choice(med2.variants), etype="MEDICATION", assertion="present", code=med2.code, system=med2.system)
        elif slot == "DOSE":
            ent = b.add_entity(rng.choice(med1.dosages), etype="DOSAGE", assertion="present")
            if med1_ent is not None:
                b.add_relation("HAS_DOSAGE", med1_ent, ent)
        elif slot == "DOSE2":
            ent = b.add_entity(rng.choice(med2.dosages), etype="DOSAGE", assertion="present")
            if med2_ent is not None:
                b.add_relation("HAS_DOSAGE", med2_ent, ent)
        elif slot == "ROUTE":
            routes = med1.routes_en if lang == "en" else med1.routes_fr
            ent = b.add_entity(rng.choice(routes), etype="ROUTE", assertion="present")
            if med1_ent is not None:
                b.add_relation("HAS_ROUTE", med1_ent, ent)
        elif slot == "ROUTE2":
            routes = med2.routes_en if lang == "en" else med2.routes_fr
            ent = b.add_entity(rng.choice(routes), etype="ROUTE", assertion="present")
            if med2_ent is not None:
                b.add_relation("HAS_ROUTE", med2_ent, ent)
        elif slot == "LAB":
            lab1_ent = b.add_entity(rng.choice(lab1.variants), etype="LAB_TEST", assertion="present", code=lab1.code, system=lab1.system)
        elif slot == "LAB2":
            lab2_ent = b.add_entity(rng.choice(lab2.variants), etype="LAB_TEST", assertion="present", code=lab2.code, system=lab2.system)
        elif slot == "VAL":
            val = round(rng.uniform(lab1.low * 0.7, lab1.high * 1.6), 1)
            ent = b.add_entity(str(val), etype="LAB_VALUE", assertion="present")
            if lab1_ent is not None:
                b.add_relation("HAS_VALUE", lab1_ent, ent)
        elif slot == "VAL2":
            val = round(rng.uniform(lab2.low * 0.7, lab2.high * 1.6), 1)
            ent = b.add_entity(str(val), etype="LAB_VALUE", assertion="present")
            if lab2_ent is not None:
                b.add_relation("HAS_VALUE", lab2_ent, ent)
        elif slot == "UNIT":
            ent = b.add_entity(lab1.unit, etype="LAB_UNIT", assertion="present")
            if lab1_ent is not None:
                b.add_relation("HAS_UNIT", lab1_ent, ent)
        elif slot == "UNIT2":
            ent = b.add_entity(lab2.unit, etype="LAB_UNIT", assertion="present")
            if lab2_ent is not None:
                b.add_relation("HAS_UNIT", lab2_ent, ent)
        elif slot == "PROC":
            ent = b.add_entity(_surface_variant(rng.choice(proc1.variants), rng), etype="PROCEDURE", assertion="present", code=proc1.code, system=proc1.system)
            if dx_pos_ent is not None:
                b.add_relation("REVEALS", ent, dx_pos_ent)
        elif slot == "ANAT":
            ent = b.add_entity(_surface_variant(rng.choice(anat1.variants), rng), etype="ANATOMY", assertion="present", code=anat1.code, system=anat1.system)
            if dx_pos_ent is not None:
                b.add_relation("LOCATED_IN", dx_pos_ent, ent)
        else:
            b.add_text(m.group(0))

        pos = m.end()

    if pos < len(pattern):
        b.add_text(pattern[pos:])


_V3_1_HEADERS: dict[str, dict[str, tuple[str, ...]]] = {
    "discharge_summary": {
        "en": (
            "DISCHARGE SUMMARY",
            "HOSPITAL DISCHARGE REPORT",
            "INPATIENT DISCHARGE NOTE",
            "SUMMARY OF HOSPITALIZATION",
        ),
        "fr": (
            "COMPTE RENDU D'HOSPITALISATION",
            "LETTRE DE SORTIE",
            "SYNTHÈSE D'HOSPITALISATION",
            "BILAN DE SORTIE",
        ),
    },
    "consultation": {
        "en": (
            "OUTPATIENT CONSULTATION NOTE",
            "SPECIALIST CONSULTATION",
            "AMBULATORY CLINIC VISIT",
            "FOLLOW-UP CONSULTATION",
        ),
        "fr": (
            "COMPTE RENDU DE CONSULTATION",
            "AVIS SPÉCIALISÉ",
            "CONSULTATION EXTERNE",
            "SUIVI AMBULATOIRE",
        ),
    },
    "lab_report": {
        "en": (
            "LABORATORY RESULTS REPORT",
            "CLINICAL BIOCHEMISTRY PANEL",
            "HEMATOLOGY & CHEMISTRY REPORT",
            "DIAGNOSTIC LAB SUMMARY",
        ),
        "fr": (
            "COMPTE RENDU DE BIOLOGIE MÉDICALE",
            "BILAN BIOLOGIQUE",
            "RÉSULTATS DE LABORATOIRE",
            "ANALYSES HÉMATOLOGIQUES ET BIOCHIMIQUES",
        ),
    },
    "prescription": {
        "en": (
            "MEDICAL PRESCRIPTION",
            "OUTPATIENT MEDICATION ORDER",
            "PHARMACY PRESCRIPTION",
            "TREATMENT ORDER",
        ),
        "fr": (
            "ORDONNANCE MÉDICALE",
            "PRESCRIPTION MÉDICAMENTEUSE",
            "ORDONNANCE DE SORTIE",
            "TRAITEMENT PRESCRIT",
        ),
    },
    "radiology_report": {
        "en": (
            "RADIOLOGY IMAGING REPORT",
            "DIAGNOSTIC IMAGING STUDY",
            "CT / MRI RADIOLOGY NOTE",
            "MEDICAL IMAGING INTERPRETATION",
        ),
        "fr": (
            "COMPTE RENDU D'IMAGERIE MÉDICALE",
            "RAPPORT DE RADIOLOGIE",
            "EXAMEN TOMODENSITOMÉTRIQUE / IRM",
            "INTERPRÉTATION RADIOLOGIQUE",
        ),
    },
    "operative_note": {
        "en": (
            "OPERATIVE PROCEDURE NOTE",
            "SURGICAL INTERVENTION REPORT",
            "POST-OPERATIVE SUMMARY",
            "OPERATING ROOM RECORD",
        ),
        "fr": (
            "COMPTE RENDU OPÉRATOIRE",
            "RAPPORT D'INTERVENTION CHIRURGICALE",
            "NOTE POST-OPÉRATOIRE",
            "PROTOCOLE OPÉRATOIRE",
        ),
    },
}

_V3_1_AMBIGUOUS_HEADERS: dict[str, tuple[str, ...]] = {
    "en": ("CLINICAL RECORD", "MEDICAL REPORT", "DIAGNOSTIC NOTE"),
    "fr": ("DOSSIER CLINIQUE", "RAPPORT MÉDICAL", "NOTE DIAGNOSTIQUE"),
}

_V3_1_SECTION_LABELS: dict[str, dict[str, dict[str, str]]] = {
    "discharge_summary": {
        "en": {
            "chief_complaint": "ADMISSION REASON:",
            "history": "PAST MEDICAL HISTORY:",
            "hpi": "HOSPITAL COURSE:",
            "labs": "DISCHARGE LABS:",
            "medications": "DISCHARGE MEDICATIONS:",
            "assessment": "DISCHARGE DIAGNOSES:",
            "plan": "DISCHARGE PLAN:",
        },
        "fr": {
            "chief_complaint": "MOTIF D'HOSPITALISATION :",
            "history": "ANTÉCÉDENTS :",
            "hpi": "ÉVOLUTION EN HOSPITALISATION :",
            "labs": "BIOLOGIE DE SORTIE :",
            "medications": "TRAITEMENT DE SORTIE :",
            "assessment": "DIAGNOSTICS DE SORTIE :",
            "plan": "CONDUITE À TENIR À LA SORTIE :",
        },
    },
    "consultation": {
        "en": {
            "chief_complaint": "REASON FOR CONSULTATION:",
            "history": "MEDICAL HISTORY:",
            "hpi": "CLINICAL EXAMINATION:",
            "medications": "CURRENT MEDICATIONS:",
            "assessment": "CLINICAL ASSESSMENT:",
            "plan": "FOLLOW-UP PLAN:",
        },
        "fr": {
            "chief_complaint": "MOTIF DE CONSULTATION :",
            "history": "ANTÉCÉDENTS :",
            "hpi": "EXAMEN CLINIQUE :",
            "medications": "TRAITEMENT EN COURS :",
            "assessment": "AVIS ET DIAGNOSTIC :",
            "plan": "SUIVI PROPOSÉ :",
        },
    },
    "lab_report": {
        "en": {"labs": "LABORATORY PANEL RESULTS:", "assessment": "BIOLOGICAL INTERPRETATION:"},
        "fr": {"labs": "RÉSULTATS BIOLOGIQUES :", "assessment": "INTERPRÉTATION BIOLOGIQUE :"},
    },
    "prescription": {
        "en": {"medications": "PRESCRIBED MEDICATIONS (RX):", "plan": "PHARMACY INSTRUCTIONS:"},
        "fr": {"medications": "PRESCRIPTION MÉDICAMENTEUSE (RP) :", "plan": "CONSIGNES DE DÉLIVRANCE :"},
    },
    "radiology_report": {
        "en": {
            "chief_complaint": "IMAGING INDICATION & TECHNIQUE:",
            "assessment": "RADIOLOGY FINDINGS & IMPRESSION:",
            "plan": "RADIOLOGIST RECOMMENDATION:",
        },
        "fr": {
            "chief_complaint": "INDICATION ET TECHNIQUE D'IMAGERIE :",
            "assessment": "RÉSULTATS RADIOLOGIQUES ET CONCLUSION :",
            "plan": "RECOMMANDATION RADIOLOGIQUE :",
        },
    },
    "operative_note": {
        "en": {
            "chief_complaint": "PREOPERATIVE INDICATION:",
            "hpi": "OPERATIVE FINDINGS:",
            "assessment": "PROCEDURE PERFORMED & COMPLICATIONS:",
            "plan": "POSTOPERATIVE ORDERS:",
        },
        "fr": {
            "chief_complaint": "INDICATION OPÉRATOIRE :",
            "hpi": "CONSTATATIONS PER-OPÉRATOIRES :",
            "assessment": "GESTE OPÉRATOIRE ET SUITES :",
            "plan": "CONSIGNES POST-OPÉRATOIRES :",
        },
    },
}


def _add_v3_1_header(
    b: _DocBuilder,
    *,
    doc_type: str,
    language: str,
    patient_id: str,
    doc_date: date,
    rng: random.Random,
) -> None:
    if rng.random() < 0.12:
        title = rng.choice(_V3_1_AMBIGUOUS_HEADERS[language])
    else:
        title = rng.choice(_V3_1_HEADERS[doc_type][language])
    date_txt, date_norm = _format_date_with_norm(doc_date, language, rng)
    b.add_text(f"{title} — Patient ")
    b.add_entity(patient_id, etype="PATIENT_ID", assertion="present")
    b.add_text(" — Date: " if language == "en" else " — Date : ")
    b.add_entity_with_timex(
        date_txt,
        etype="DATE",
        ttype="DATE",
        normalized=date_norm,
        assertion="present",
    )


def _v3_1_sections_for_doc_type(doc_type: str, rng: random.Random) -> list[tuple[str, int]]:
    """Return [(section_name, n_sentences), ...] tailored to each document type."""
    if doc_type == "discharge_summary":
        secs: list[tuple[str, int]] = [
            ("chief_complaint", 1),
            ("history", 1),
            ("hpi", rng.choice([2, 3])),
            ("labs", rng.choice([1, 2])),
            ("medications", rng.choice([2, 3])),
            ("assessment", rng.choice([1, 2])),
            ("plan", rng.choice([1, 2])),
        ]
        if rng.random() < 0.3:
            mid = secs[1:-1]
            rng.shuffle(mid)
            secs = [secs[0]] + mid + [secs[-1]]
        return secs
    if doc_type == "consultation":
        secs = [
            ("chief_complaint", 1),
            ("history", 1),
            ("hpi", rng.choice([1, 2])),
            ("assessment", rng.choice([1, 2])),
            ("plan", 1),
        ]
        if rng.random() < 0.35:
            secs.insert(3, ("medications", 1))
        return secs
    if doc_type == "lab_report":
        secs = [("labs", rng.choice([3, 4, 5]))]
        if rng.random() < 0.4:
            secs.append(("assessment", 1))
        return secs
    if doc_type == "prescription":
        secs = [("medications", rng.choice([3, 4, 5]))]
        if rng.random() < 0.4:
            secs.append(("plan", 1))
        return secs
    if doc_type == "radiology_report":
        return [
            ("chief_complaint", 1),
            ("assessment", rng.choice([2, 3])),
            ("plan", 1),
        ]
    # operative_note
    return [
        ("chief_complaint", 1),
        ("hpi", rng.choice([1, 2])),
        ("assessment", rng.choice([2, 3])),
        ("plan", 1),
    ]


def _add_v3_1_x8_generalization_sentences(
    b: _DocBuilder,
    *,
    pool: _LexiconPool,
    language: str,
    rng: random.Random,
) -> None:
    """Add diverse train/val-only prose sentences (pronouns, relatives, N-year-old ages, hospitalisation, tid/bid/qd/3x/j, excluded-by-<test>)."""
    dx = rng.choice(pool.diagnoses)
    sx = rng.choice(pool.symptoms)
    med = rng.choice(pool.medications)
    proc = rng.choice(pool.procedures)
    rel_names = ("Ibrahima Fall", "Awa Diop", "Moussa Ndiaye", "Claire Martin", "David Brooks", "Sarah Patel")
    rel = rng.choice(rel_names)
    age_n = rng.randint(4, 88)

    if language == "en":
        freq_txt, freq_norm = rng.choice([("tid", "TID"), ("bid", "BID"), ("qd", "QD"), ("qid", "QID")])
        dose_txt = rng.choice(med.dosages) if med.dosages else "10 mg"
        choice = rng.randrange(4)
        b.add_text("\n")
        if choice == 0:
            b.add_text(f"Seen with her son {rel}; ")
            m_ent = b.add_entity(med.variants[0].lower(), etype="MEDICATION", code=med.concept_id)
            b.add_text(" ")
            d_ent = b.add_entity(dose_txt, etype="DOSAGE")
            b.add_text(" ")
            f_ent, _ = b.add_entity_with_timex(freq_txt, etype="FREQUENCY", ttype="FREQUENCY", normalized=freq_norm)
            b.add_text(" for 7 days.")
            b.add_relation("HAS_DOSAGE", m_ent, d_ent)
            b.add_relation("HAS_FREQUENCY", m_ent, f_ent)
        elif choice == 1:
            b.add_text(f"A {age_n}-year-old patient with ")
            b.add_entity(rng.choice(sx.variants), etype="SYMPTOM", code=sx.concept_id, assertion="present")
            b.add_text(", hospitalized for 2 days.")
        elif choice == 2:
            b.add_entity(rng.choice(dx.variants), etype="DIAGNOSIS", code=dx.concept_id, assertion="absent")
            b.add_text(" was excluded on ")
            b.add_entity(rng.choice(proc.variants), etype="PROCEDURE", code=proc.concept_id, assertion="present")
            b.add_text(".")
        else:
            if rng.random() < 0.25:
                b.add_entity("Graves disease", etype="DIAGNOSIS", code="E05.00", assertion="present")
                b.add_text(" treated with ")
                b.add_entity("methimazole", etype="MEDICATION", code="H03BB02", assertion="present")
                b.add_text(".")
            else:
                b.add_entity(rng.choice(dx.variants), etype="DIAGNOSIS", code=dx.concept_id, assertion="present")
                b.add_text(" treated with ")
                b.add_entity(med.variants[0].lower(), etype="MEDICATION", code=med.concept_id, assertion="present")
                b.add_text(".")
    else:
        freq_txt, freq_norm = rng.choice([("3×/j", "TID"), ("2×/j", "BID"), ("1×/j", "QD"), ("tid", "TID")])
        dose_txt = rng.choice(med.dosages) if med.dosages else "10 mg"
        choice = rng.randrange(4)
        b.add_text("\n")
        if choice == 0:
            b.add_text(f"Accompagné par son fils {rel} ; ")
            m_ent = b.add_entity(med.variants[0].lower(), etype="MEDICATION", code=med.concept_id)
            b.add_text(" ")
            d_ent = b.add_entity(dose_txt, etype="DOSAGE")
            b.add_text(" ")
            f_ent, _ = b.add_entity_with_timex(freq_txt, etype="FREQUENCY", ttype="FREQUENCY", normalized=freq_norm)
            b.add_text(" pendant 7 jours.")
            b.add_relation("HAS_DOSAGE", m_ent, d_ent)
            b.add_relation("HAS_FREQUENCY", m_ent, f_ent)
        elif choice == 1:
            sx_txt = "céphalée" if rng.random() < 0.25 else rng.choice(sx.variants)
            sx_code = "R51.9" if sx_txt == "céphalée" else sx.concept_id
            b.add_text(f"Patiente âgée de {age_n} ans, hospitalisée depuis 2 jours pour ")
            b.add_entity(sx_txt, etype="SYMPTOM", code=sx_code, assertion="present")
            b.add_text(".")
        elif choice == 2:
            b.add_entity(rng.choice(dx.variants), etype="DIAGNOSIS", code=dx.concept_id, assertion="absent")
            b.add_text(" écartée par ")
            b.add_entity(rng.choice(proc.variants), etype="PROCEDURE", code=proc.concept_id, assertion="present")
            b.add_text(".")
        else:
            b.add_entity(rng.choice(dx.variants), etype="DIAGNOSIS", code=dx.concept_id, assertion="present")
            b.add_text(" traité par ")
            b.add_entity(med.variants[0].lower(), etype="MEDICATION", code=med.concept_id, assertion="present")
            b.add_text(".")


def _generate_v3_document(
    *,
    doc_id: str,
    patient_id: str,
    language: str,
    doc_type: str,
    split_name: str,
    pool: _LexiconPool,
    tpl_map: dict[tuple[str, str], list[TemplateSpec]],
    rng: random.Random,
    version: str = "v3",
) -> AnnotatedDocument:
    b = _DocBuilder()
    doc_date = date(2025, 1, 1) + timedelta(days=rng.randint(0, 480))

    if version == "v3_1":
        _add_v3_1_header(
            b,
            doc_type=doc_type,
            language=language,
            patient_id=patient_id,
            doc_date=doc_date,
            rng=rng,
        )
        labels_map = _V3_1_SECTION_LABELS[doc_type][language]
        for sec, n_sents in _v3_1_sections_for_doc_type(doc_type, rng):
            b.add_text("\n\n" if rng.random() < 0.7 else "\n")
            label = labels_map.get(sec)
            if label:
                b.add_text(f"{label} ")
            candidates = tpl_map[(language, sec)]
            chosen_tpls = rng.sample(candidates, k=min(n_sents, len(candidates)))
            for s_i, tpl in enumerate(chosen_tpls):
                if s_i > 0:
                    b.add_text("\n" if sec in {"medications", "labs"} else " ")
                _render_template(b, tpl.pattern, pool, language, patient_id, doc_date, rng)
        if split_name == "train":
            import zlib

            doc_idx = int(doc_id.rsplit("_", 1)[-1]) if doc_id.rsplit("_", 1)[-1].isdigit() else 999
            if doc_idx > 60:
                h_val = zlib.crc32(f"{doc_id}:{split_name}:{language}".encode("utf-8"))
                if (h_val % 100) < 45:
                    _add_v3_1_x8_generalization_sentences(b, pool=pool, language=language, rng=random.Random(h_val))
        return AnnotatedDocument(
            doc_id=doc_id,
            patient_id=patient_id,
            language=language,
            doc_type=doc_type,
            document_date=doc_date.isoformat(),
            source="synthetic_v3_1",
            text=b.build_text(),
            entities=b.entities,
            relations=b.relations,
            timex=b.timex,
            split=split_name,
            metadata={"doc_date": doc_date.isoformat(), "generator": "v3_1"},
        )

    # Pick structure: short note (missing optional sections) vs standard vs long multi-sentence note
    length_mode = rng.choices(["short", "standard", "long"], weights=[0.25, 0.50, 0.25])[0]
    body_sections = ["chief_complaint", "history", "hpi", "medications", "labs", "assessment"]
    if length_mode == "short":
        # Omit 2 optional sections
        omit = set(rng.sample(["history", "labs", "medications"], k=2))
        body_sections = [s for s in body_sections if s not in omit]
    if rng.random() < 0.45:
        rng.shuffle(body_sections)

    ordered_sections = ["header"] + body_sections + ["plan"]

    for idx, sec in enumerate(ordered_sections):
        if idx > 0:
            b.add_text("\n\n" if rng.random() < 0.7 else "\n")
        candidates = tpl_map[(language, sec)]
        n_sents = 1
        if sec in {"hpi", "medications", "labs", "assessment"}:
            n_sents = 2 if length_mode == "standard" and rng.random() < 0.5 else (3 if length_mode == "long" else 1)
        chosen = [rng.choice(candidates) for _ in range(n_sents)]
        for s_i, tpl in enumerate(chosen):
            if s_i > 0:
                b.add_text("\n" if sec in {"medications", "labs"} else " ")
            _render_template(b, tpl.pattern, pool, language, patient_id, doc_date, rng)

    return AnnotatedDocument(
        doc_id=doc_id,
        patient_id=patient_id,
        language=language,
        doc_type=doc_type,
        document_date=doc_date.isoformat(),
        source="synthetic_v3",
        text=b.build_text(),
        entities=b.entities,
        relations=b.relations,
        timex=b.timex,
        split=split_name,
        metadata={"doc_date": doc_date.isoformat(), "generator": "v3"},
    )


def generate_v3_corpus_splits(
    *,
    split_counts_per_lang: dict[str, int] | None = None,
    seed: int = 42,
    version: str = "v3",
) -> dict[str, list[AnnotatedDocument]]:
    """Generate SYNTH-MED-V3 / V3.1 splits: train, val, test_iid, test_unseen_terms, test_unseen_templates."""
    counts = split_counts_per_lang or {
        "train": 300,
        "val": 80,
        "test_iid": 80,
        "test_unseen_terms": 80,
        "test_unseen_templates": 80,
    }
    all_tpls = build_all_templates()
    seen_tpls, unseen_tpls = partition_templates(all_tpls, unseen_ratio=0.30, seed=seed)

    pools = {
        ("en", False): _LexiconPool("en", use_unseen=False, seed=seed),
        ("en", True): _LexiconPool("en", use_unseen=True, seed=seed),
        ("fr", False): _LexiconPool("fr", use_unseen=False, seed=seed),
        ("fr", True): _LexiconPool("fr", use_unseen=True, seed=seed),
    }

    rng = random.Random(seed)
    out: dict[str, list[AnnotatedDocument]] = {}
    patient_counter = 1000

    for split_name, n_per_lang in counts.items():
        use_unseen_terms = split_name == "test_unseen_terms"
        use_unseen_tpls = split_name == "test_unseen_templates"
        tpl_map = unseen_tpls if use_unseen_tpls else seen_tpls

        docs: list[AnnotatedDocument] = []
        for lang in ("en", "fr"):
            pool = pools[(lang, use_unseen_terms)]
            for i in range(1, n_per_lang + 1):
                patient_counter += 1
                pid = f"PAT-{lang.upper()}-{patient_counter:05d}"
                doc_id = f"{version}_{split_name}_{lang}_{i:04d}"
                doc_type = DOC_TYPES[(i - 1) % len(DOC_TYPES)]
                docs.append(
                    _generate_v3_document(
                        doc_id=doc_id,
                        patient_id=pid,
                        language=lang,
                        doc_type=doc_type,
                        split_name=split_name,
                        pool=pool,
                        tpl_map=tpl_map,
                        rng=rng,
                        version=version,
                    )
                )
        out[split_name] = docs

    return out


def generate_v3_flat_list(
    *,
    split_counts_per_lang: dict[str, int] | None = None,
    seed: int = 42,
    version: str = "v3",
) -> list[AnnotatedDocument]:
    splits = generate_v3_corpus_splits(split_counts_per_lang=split_counts_per_lang, seed=seed, version=version)
    flat: list[AnnotatedDocument] = []
    for docs in splits.values():
        flat.extend(docs)
    return flat


def summarize_v3_resources() -> dict[str, Any]:
    """Return exact lexicon & template counts for audit/tests."""
    tpls = build_all_templates()
    return {
        "diagnoses_en": len(DIAGNOSES_EN),
        "diagnoses_fr": len(DIAGNOSES_FR),
        "symptoms_en": len(SYMPTOMS_EN),
        "symptoms_fr": len(SYMPTOMS_FR),
        "medications_en": len(MEDICATIONS_EN),
        "medications_fr": len(MEDICATIONS_FR),
        "labs_en": len(LABS_EN),
        "labs_fr": len(LABS_FR),
        "procedures_en": len(PROCEDURES_EN),
        "procedures_fr": len(PROCEDURES_FR),
        "anatomy_en": len(ANATOMY_EN),
        "anatomy_fr": len(ANATOMY_FR),
        "total_templates": len(tpls),
    }
