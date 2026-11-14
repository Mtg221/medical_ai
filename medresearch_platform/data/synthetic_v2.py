"""Synthetic bilingual (EN/FR) medical corpus generator v2 with exact gold spans by construction."""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import docx
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as pdf_canvas

from medresearch_platform.data.schema import (
    DOC_TYPES,
    AnnotatedDocument,
    EntitySpan,
    RelationSpan,
    TimexSpan,
)


@dataclass(frozen=True)
class LexItem:
    en: str
    fr: str
    code: str
    system: str


SYMPTOMS: tuple[LexItem, ...] = (
    LexItem("shortness of breath", "dyspnée", "R06.0", "ICD-10"),
    LexItem("chest pain", "douleur thoracique", "R07.9", "ICD-10"),
    LexItem("fever", "fièvre", "R50.9", "ICD-10"),
    LexItem("dry cough", "toux sèche", "R05", "ICD-10"),
    LexItem("productive cough", "toux productive", "R05", "ICD-10"),
    LexItem("fatigue", "asthénie", "R53", "ICD-10"),
    LexItem("dizziness", "vertiges", "R42", "ICD-10"),
    LexItem("nausea", "nausées", "R11", "ICD-10"),
    LexItem("headache", "céphalées", "R51", "ICD-10"),
    LexItem("abdominal pain", "douleur abdominale", "R10.4", "ICD-10"),
    LexItem("palpitations", "palpitations", "R00.2", "ICD-10"),
    LexItem("lower limb edema", "œdème des membres inférieurs", "R60.0", "ICD-10"),
)

DIAGNOSES: tuple[LexItem, ...] = (
    LexItem("essential hypertension", "hypertension artérielle essentielle", "I10", "ICD-10"),
    LexItem("type 2 diabetes mellitus", "diabète de type 2", "E11.9", "ICD-10"),
    LexItem("bacterial pneumonia", "pneumopathie bactérienne", "J15.9", "ICD-10"),
    LexItem("pulmonary embolism", "embolie pulmonaire", "I26.9", "ICD-10"),
    LexItem("atrial fibrillation", "fibrillation auriculaire", "I48.91", "ICD-10"),
    LexItem("acute bronchitis", "bronchite aiguë", "J20.9", "ICD-10"),
    LexItem("heart failure", "insuffisance cardiaque", "I50.9", "ICD-10"),
    LexItem("chronic kidney disease", "maladie rénale chronique", "N18.9", "ICD-10"),
    LexItem("asthma", "asthme", "J45.9", "ICD-10"),
    LexItem("urinary tract infection", "infection urinaire", "N39.0", "ICD-10"),
    LexItem("ischemic stroke", "accident vasculaire cérébral ischémique", "I63.9", "ICD-10"),
    LexItem("deep vein thrombosis", "thrombose veineuse profonde", "I82.40", "ICD-10"),
)

MEDICATIONS: tuple[tuple[LexItem, str, str, str, str, str], ...] = (
    (LexItem("metformin", "metformine", "A10BA02", "ATC"), "500 mg", "twice daily", "deux fois par jour", "orally", "par voie orale"),
    (LexItem("amlodipine", "amlodipine", "C08CA01", "ATC"), "5 mg", "once daily", "une fois par jour", "orally", "par voie orale"),
    (LexItem("amoxicillin", "amoxicilline", "J01CA04", "ATC"), "1000 mg", "three times daily", "trois fois par jour", "orally", "par voie orale"),
    (LexItem("apixaban", "apixaban", "B01AF02", "ATC"), "5 mg", "twice daily", "deux fois par jour", "orally", "par voie orale"),
    (LexItem("furosemide", "furosémide", "C03CA01", "ATC"), "40 mg", "once daily", "une fois par jour", "intravenously", "par voie intraveineuse"),
    (LexItem("paracetamol", "paracétamol", "N02BE01", "ATC"), "1000 mg", "every 6 hours", "toutes les 6 heures", "orally", "par voie orale"),
    (LexItem("bisoprolol", "bisoprolol", "C07AB07", "ATC"), "2.5 mg", "once daily", "une fois par jour", "orally", "par voie orale"),
    (LexItem("atorvastatin", "atorvastatine", "C10AA05", "ATC"), "20 mg", "at bedtime", "au coucher", "orally", "par voie orale"),
    (LexItem("enoxaparin", "énoxaparine", "B01AB05", "ATC"), "40 mg", "once daily", "une fois par jour", "subcutaneously", "par voie sous-cutanée"),
    (LexItem("salbutamol", "salbutamol", "R03AC02", "ATC"), "100 mcg", "as needed", "si besoin", "inhaled", "par inhalation"),
)

LAB_TESTS: tuple[tuple[LexItem, float, float, str], ...] = (
    (LexItem("HbA1c", "HbA1c", "4548-4", "LOINC"), 5.4, 9.6, "%"),
    (LexItem("CRP", "CRP", "1988-5", "LOINC"), 2.1, 85.0, "mg/L"),
    (LexItem("serum creatinine", "créatininémie", "2160-0", "LOINC"), 62.0, 185.0, "µmol/L"),
    (LexItem("hemoglobin", "hémoglobine", "718-7", "LOINC"), 9.5, 16.2, "g/dL"),
    (LexItem("D-dimer", "D-dimères", "48065-7", "LOINC"), 180.0, 1950.0, "ng/mL"),
    (LexItem("troponin I", "troponine I", "10839-9", "LOINC"), 0.01, 2.40, "ng/mL"),
    (LexItem("fasting glucose", "glycémie à jeun", "1558-6", "LOINC"), 4.2, 11.8, "mmol/L"),
    (LexItem("leukocytes", "leucocytes", "6690-2", "LOINC"), 4.5, 18.4, "G/L"),
)

PROCEDURES: tuple[LexItem, ...] = (
    LexItem("chest CT", "angioscanner thoracique", "P01", "LOCAL"),
    LexItem("transthoracic echocardiography", "échocardiographie transthoracique", "P02", "LOCAL"),
    LexItem("electrocardiogram", "électrocardiogramme", "P03", "LOCAL"),
    LexItem("chest X-ray", "radiographie thoracique", "P04", "LOCAL"),
    LexItem("abdominal ultrasound", "échographie abdominale", "P05", "LOCAL"),
    LexItem("bronchoscopy", "bronchoscopie", "P06", "LOCAL"),
)

ANATOMIES: tuple[LexItem, ...] = (
    LexItem("lungs", "poumons", "A01", "LOCAL"),
    LexItem("left lower lobe", "lobe inférieur gauche", "A02", "LOCAL"),
    LexItem("right ventricle", "ventricule droit", "A03", "LOCAL"),
    LexItem("chest", "thorax", "A04", "LOCAL"),
    LexItem("abdomen", "abdomen", "A05", "LOCAL"),
    LexItem("lower limbs", "membres inférieurs", "A06", "LOCAL"),
)

SECTION_HEADERS: dict[str, dict[str, str]] = {
    "en": {
        "header": "CLINICAL RECORD",
        "cc": "CHIEF COMPLAINT:",
        "pmh": "PAST MEDICAL HISTORY:",
        "meds": "MEDICATIONS:",
        "labs": "LABORATORY RESULTS:",
        "proc": "IMAGING & PROCEDURES:",
        "plan": "ASSESSMENT AND PLAN:",
    },
    "fr": {
        "header": "COMPTE RENDU CLINIQUE",
        "cc": "MOTIF DE CONSULTATION :",
        "pmh": "ANTÉCÉDENTS :",
        "meds": "TRAITEMENTS :",
        "labs": "BIOLOGIE :",
        "proc": "IMAGERIE ET ACTES :",
        "plan": "CONCLUSION ET CONDUITE À TENIR :",
    },
}


class _DocBuilder:
    """Incrementally constructs document text while recording exact character offsets."""

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._cursor = 0
        self._ent_seq = 0
        self._rel_seq = 0
        self._timex_seq = 0
        self.entities: list[EntitySpan] = []
        self.relations: list[RelationSpan] = []
        self.timex: list[TimexSpan] = []

    def add_text(self, s: str) -> None:
        self._parts.append(s)
        self._cursor += len(s)

    def add_entity(
        self,
        text: str,
        *,
        etype: str,
        assertion: str = "present",
        temporality: str = "current",
        code: str | None = None,
        system: str | None = None,
    ) -> EntitySpan:
        self._ent_seq += 1
        start = self._cursor
        self.add_text(text)
        end = self._cursor
        ent = EntitySpan(
            id=f"e{self._ent_seq}",
            type=etype,
            start=start,
            end=end,
            text=text,
            assertion=assertion,
            temporality=temporality,
            normalized_code=code,
            normalized_system=system,
        )
        self.entities.append(ent)
        return ent

    def add_timex(self, text: str, *, ttype: str, normalized: str | None = None) -> TimexSpan:
        self._timex_seq += 1
        start = self._cursor
        self.add_text(text)
        end = self._cursor
        tx = TimexSpan(
            id=f"t{self._timex_seq}",
            type=ttype,
            start=start,
            end=end,
            text=text,
            normalized=normalized,
        )
        self.timex.append(tx)
        return tx

    def add_relation(self, rtype: str, head: EntitySpan, tail: EntitySpan) -> None:
        self._rel_seq += 1
        self.relations.append(RelationSpan(id=f"r{self._rel_seq}", type=rtype, head=head.id, tail=tail.id))

    def build_text(self) -> str:
        return "".join(self._parts)


def _pick_lex(item: LexItem, language: str) -> str:
    return item.fr if language == "fr" else item.en


def _maybe_filler_typo(phrase: str, rng: random.Random, typo_rate: float) -> str:
    if typo_rate <= 0 or rng.random() > typo_rate:
        return phrase
    words = phrase.split(" ")
    candidates = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
    if not candidates:
        return phrase
    idx = rng.choice(candidates)
    w = words[idx]
    pos = rng.randrange(1, len(w) - 1)
    words[idx] = w[:pos] + w[pos + 1] + w[pos] + w[pos + 2 :]
    return " ".join(words)


def generate_synthetic_document(
    *,
    doc_index: int,
    patient_index: int,
    language: str,
    doc_type: str,
    rng: random.Random,
    typo_rate: float = 0.05,
) -> AnnotatedDocument:
    """Generate one bilingual synthetic clinical document with exact gold spans."""
    if language not in ("en", "fr"):
        raise ValueError(f"Unsupported language {language!r}; expected 'en' or 'fr'")

    doc_date = date(2026, 1, 1) + timedelta(days=rng.randint(0, 300))
    doc_date_iso = doc_date.isoformat()
    rel_days = rng.choice([2, 3, 5, 7, 14])
    rel_date_iso = (doc_date - timedelta(days=rel_days)).isoformat()
    patient_id = f"PAT-{language.upper()}-{patient_index:04d}"
    doc_id = f"synth_{language}_{doc_index:04d}"
    hdr = SECTION_HEADERS[language]

    sym1, sym2, sym_hyp = rng.sample(SYMPTOMS, 3)
    dx_hist, dx_fam, dx_neg, dx_pos = rng.sample(DIAGNOSES, 4)
    med1, med2 = rng.sample(MEDICATIONS, 2)
    lab1, lab2 = rng.sample(LAB_TESTS, 2)
    proc = rng.choice(PROCEDURES)
    anat = rng.choice(ANATOMIES)

    b = _DocBuilder()

    # Header
    b.add_text(f"{hdr['header']} ({doc_type.replace('_', ' ').upper()}) | ID: ")
    b.add_entity(patient_id, etype="PATIENT_ID")
    b.add_text(" | Date: ")
    b.add_entity(doc_date_iso, etype="DATE")
    b.add_timex(doc_date_iso, ttype="DATE", normalized=doc_date_iso)
    # Note: keep DATE entity and Timex without overlapping duplicate if needed; both share exact offsets cleanly.
    b.add_text("\n")

    # Build sections and optionally shuffle middle sections for structural variety
    def sec_cc() -> None:
        b.add_text(f"{hdr['cc']}\n")
        if language == "en":
            b.add_text(_maybe_filler_typo("Patient presents with ", rng, typo_rate))
            s1 = b.add_entity(_pick_lex(sym1, language), etype="SYMPTOM", code=sym1.code, system=sym1.system)
            b.add_text(" and ")
            b.add_entity(_pick_lex(sym2, language), etype="SYMPTOM", code=sym2.code, system=sym2.system)
            b.add_text(" starting ")
            rel_txt = f"{rel_days} days ago"
            b.add_entity(rel_txt, etype="DURATION")
            b.add_timex(rel_txt, ttype="RELATIVE", normalized=rel_date_iso)
            b.add_text(" affecting the ")
            a_ent = b.add_entity(_pick_lex(anat, language), etype="ANATOMY", code=anat.code, system=anat.system)
            b.add_relation("LOCATED_IN", s1, a_ent)
            b.add_text(".\n")
        else:
            b.add_text(_maybe_filler_typo("Patient admis pour ", rng, typo_rate))
            s1 = b.add_entity(_pick_lex(sym1, language), etype="SYMPTOM", code=sym1.code, system=sym1.system)
            b.add_text(" et ")
            b.add_entity(_pick_lex(sym2, language), etype="SYMPTOM", code=sym2.code, system=sym2.system)
            b.add_text(" évoluant depuis ")
            rel_txt = f"il y a {rel_days} jours"
            b.add_entity(rel_txt, etype="DURATION")
            b.add_timex(rel_txt, ttype="RELATIVE", normalized=rel_date_iso)
            b.add_text(" au niveau des ")
            a_ent = b.add_entity(_pick_lex(anat, language), etype="ANATOMY", code=anat.code, system=anat.system)
            b.add_relation("LOCATED_IN", s1, a_ent)
            b.add_text(".\n")

    def sec_pmh() -> None:
        b.add_text(f"{hdr['pmh']}\n")
        if language == "en":
            b.add_text("History of ")
            b.add_entity(
                _pick_lex(dx_hist, language),
                etype="DIAGNOSIS",
                assertion="historical",
                temporality="past",
                code=dx_hist.code,
                system=dx_hist.system,
            )
            b.add_text(". Family history of ")
            b.add_entity(
                _pick_lex(dx_fam, language),
                etype="DIAGNOSIS",
                assertion="family",
                temporality="past",
                code=dx_fam.code,
                system=dx_fam.system,
            )
            b.add_text(".\n")
        else:
            b.add_text("Antécédent de ")
            b.add_entity(
                _pick_lex(dx_hist, language),
                etype="DIAGNOSIS",
                assertion="historical",
                temporality="past",
                code=dx_hist.code,
                system=dx_hist.system,
            )
            b.add_text(". Antécédents familiaux de ")
            b.add_entity(
                _pick_lex(dx_fam, language),
                etype="DIAGNOSIS",
                assertion="family",
                temporality="past",
                code=dx_fam.code,
                system=dx_fam.system,
            )
            b.add_text(".\n")

    def sec_meds() -> None:
        b.add_text(f"{hdr['meds']}\n")
        m_item, dose, freq_en, freq_fr, route_en, route_fr = med1
        freq = freq_fr if language == "fr" else freq_en
        route = route_fr if language == "fr" else route_en
        dur = "pendant 7 jours" if language == "fr" else "for 7 days"
        prefix = "Traitement en cours : " if language == "fr" else "Current therapy: "
        b.add_text(prefix)
        m_ent = b.add_entity(_pick_lex(m_item, language), etype="MEDICATION", code=m_item.code, system=m_item.system)
        b.add_text(" ")
        d_ent = b.add_entity(dose, etype="DOSAGE")
        b.add_text(" ")
        f_ent = b.add_entity(freq, etype="FREQUENCY")
        b.add_timex(freq, ttype="FREQUENCY")
        b.add_text(" ")
        r_ent = b.add_entity(route, etype="ROUTE")
        b.add_text(" ")
        b.add_entity(dur, etype="DURATION")
        b.add_timex(dur, ttype="DURATION", normalized="P7D")
        b.add_relation("HAS_DOSAGE", m_ent, d_ent)
        b.add_relation("HAS_FREQUENCY", m_ent, f_ent)
        b.add_relation("HAS_ROUTE", m_ent, r_ent)
        b.add_text(".\n")

    def sec_labs() -> None:
        b.add_text(f"{hdr['labs']}\n")
        for idx, (l_item, lo, hi, unit) in enumerate((lab1, lab2)):
            val = f"{rng.uniform(lo, hi):.1f}"
            if idx > 0:
                b.add_text("; ")
            lt_ent = b.add_entity(_pick_lex(l_item, language), etype="LAB_TEST", code=l_item.code, system=l_item.system)
            b.add_text(": ")
            lv_ent = b.add_entity(val, etype="LAB_VALUE")
            b.add_text(" ")
            lu_ent = b.add_entity(unit, etype="LAB_UNIT")
            b.add_relation("HAS_VALUE", lt_ent, lv_ent)
            b.add_relation("HAS_UNIT", lv_ent, lu_ent)
        b.add_text(".\n")

    def sec_proc_and_plan() -> None:
        b.add_text(f"{hdr['proc']}\n")
        p_ent = b.add_entity(_pick_lex(proc, language), etype="PROCEDURE", code=proc.code, system=proc.system)
        if language == "en":
            b.add_text(" performed; ")
            dx_n = b.add_entity(
                _pick_lex(dx_neg, language),
                etype="DIAGNOSIS",
                assertion="absent",
                code=dx_neg.code,
                system=dx_neg.system,
            )
            b.add_text(" ruled out.\n")
            b.add_text(f"{hdr['plan']}\n")
            b.add_text("Suspected ")
            dx_p = b.add_entity(
                _pick_lex(dx_pos, language),
                etype="DIAGNOSIS",
                assertion="possible",
                code=dx_pos.code,
                system=dx_pos.system,
            )
            b.add_relation("REVEALS", p_ent, dx_n)
            b.add_text(". Prescribed ")
            m2_item, m2_dose, _, _, _, _ = med2
            m2_ent = b.add_entity(
                _pick_lex(m2_item, language),
                etype="MEDICATION",
                temporality="planned",
                code=m2_item.code,
                system=m2_item.system,
            )
            b.add_text(" ")
            d2_ent = b.add_entity(m2_dose, etype="DOSAGE", temporality="planned")
            b.add_relation("HAS_DOSAGE", m2_ent, d2_ent)
            b.add_relation("TREATS", m2_ent, dx_p)
            b.add_text(". Return if ")
            b.add_entity(
                _pick_lex(sym_hyp, language),
                etype="SYMPTOM",
                assertion="hypothetical",
                temporality="planned",
                code=sym_hyp.code,
                system=sym_hyp.system,
            )
            b.add_text(" develops.\n")
        else:
            b.add_text(" réalisé ; absence d'")
            dx_n = b.add_entity(
                _pick_lex(dx_neg, language),
                etype="DIAGNOSIS",
                assertion="absent",
                code=dx_neg.code,
                system=dx_neg.system,
            )
            b.add_text(".\n")
            b.add_text(f"{hdr['plan']}\n")
            b.add_text("Suspicion de ")
            dx_p = b.add_entity(
                _pick_lex(dx_pos, language),
                etype="DIAGNOSIS",
                assertion="possible",
                code=dx_pos.code,
                system=dx_pos.system,
            )
            b.add_relation("REVEALS", p_ent, dx_n)
            b.add_text(". Prescription de ")
            m2_item, m2_dose, _, _, _, _ = med2
            m2_ent = b.add_entity(
                _pick_lex(m2_item, language),
                etype="MEDICATION",
                temporality="planned",
                code=m2_item.code,
                system=m2_item.system,
            )
            b.add_text(" ")
            d2_ent = b.add_entity(m2_dose, etype="DOSAGE", temporality="planned")
            b.add_relation("HAS_DOSAGE", m2_ent, d2_ent)
            b.add_relation("TREATS", m2_ent, dx_p)
            b.add_text(". Consulter si apparition de ")
            b.add_entity(
                _pick_lex(sym_hyp, language),
                etype="SYMPTOM",
                assertion="hypothetical",
                temporality="planned",
                code=sym_hyp.code,
                system=sym_hyp.system,
            )
            b.add_text(".\n")

    middle_sections = [sec_cc, sec_pmh, sec_meds, sec_labs]
    rng.shuffle(middle_sections)
    for fn in middle_sections:
        fn()
    sec_proc_and_plan()

    # Deduplicate overlapping DATE entity vs timex: keep both cleanly, validate spans
    doc = AnnotatedDocument(
        doc_id=doc_id,
        patient_id=patient_id,
        language=language,
        doc_type=doc_type,
        document_date=doc_date_iso,
        source="synthetic_v2",
        text=b.build_text(),
        entities=b.entities,
        relations=b.relations,
        timex=b.timex,
        metadata={"synthetic": True, "generator_version": "2.0.0"},
    )
    errs = doc.validate_spans()
    if errs:
        raise RuntimeError(f"Generator bug produced invalid spans: {errs[:3]}")
    return doc


def generate_synthetic_corpus(
    *,
    en_count: int = 400,
    fr_count: int = 400,
    seed: int = 42,
    typo_rate: float = 0.05,
) -> list[AnnotatedDocument]:
    """Generate a balanced EN+FR synthetic corpus across all 6 document types."""
    rng = random.Random(seed)
    docs: list[AnnotatedDocument] = []
    patient_counter = 1
    doc_counter = 1

    for lang, count in (("en", en_count), ("fr", fr_count)):
        for i in range(count):
            doc_type = DOC_TYPES[i % len(DOC_TYPES)]
            # ~80% unique patients, ~20% longitudinal follow-up for same patient
            if i > 0 and rng.random() < 0.20:
                pid_idx = max(1, patient_counter - rng.randint(1, 3))
            else:
                pid_idx = patient_counter
                patient_counter += 1
            docs.append(
                generate_synthetic_document(
                    doc_index=doc_counter,
                    patient_index=pid_idx,
                    language=lang,
                    doc_type=doc_type,
                    rng=rng,
                    typo_rate=typo_rate,
                )
            )
            doc_counter += 1
    return docs


def render_degraded_scan_png(text: str, *, seed: int = 42) -> bytes:
    """Render clinical text to a degraded scan PNG (rotation, blur, Gaussian noise, JPEG artifacts)."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    lines = text.splitlines()[:28]
    width, height = 1000, max(400, 48 + len(lines) * 26)
    img = Image.new("RGB", (width, height), color=(248, 246, 240))
    draw = ImageDraw.Draw(img)
    y = 24
    for line in lines:
        draw.text((28, y), line[:105], fill=(24, 24, 28))
        y += 24

    # Slight scan skew & blur
    angle = rng.uniform(-2.0, 2.0)
    img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(244, 242, 236))
    img = img.filter(ImageFilter.GaussianBlur(radius=0.55))

    # Add subtle scanner noise
    arr = np.asarray(img, dtype=np.float32)
    noise = np_rng.normal(0.0, 6.0, size=arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    noisy = Image.fromarray(arr)

    # JPEG artifact pass then PNG output
    jpg_buf = io.BytesIO()
    noisy.save(jpg_buf, format="JPEG", quality=62)
    jpg_buf.seek(0)
    with Image.open(jpg_buf) as reloaded:
        png_buf = io.BytesIO()
        reloaded.save(png_buf, format="PNG")
        return png_buf.getvalue()


def render_document_artifacts(doc: AnnotatedDocument, out_dir: Path, *, seed: int = 42) -> dict[str, str]:
    """Render a document to PDF, DOCX, and degraded scan PNG in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    # PDF
    pdf_path = out_dir / f"{doc.doc_id}.pdf"
    pdf_buf = io.BytesIO()
    c = pdf_canvas.Canvas(pdf_buf, pagesize=letter)
    y = 750
    for line in doc.text.splitlines():
        c.drawString(40, y, line[:100])
        y -= 16
        if y < 40:
            c.showPage()
            y = 750
    c.save()
    pdf_path.write_bytes(pdf_buf.getvalue())
    paths["pdf"] = str(pdf_path)

    # DOCX
    docx_path = out_dir / f"{doc.doc_id}.docx"
    d = docx.Document()
    for line in doc.text.splitlines():
        d.add_paragraph(line)
    d.save(str(docx_path))
    paths["docx"] = str(docx_path)

    # Degraded scan
    scan_path = out_dir / f"{doc.doc_id}_scan.png"
    scan_path.write_bytes(render_degraded_scan_png(doc.text, seed=seed))
    paths["degraded_scan"] = str(scan_path)
    return paths
