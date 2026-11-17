"""Step 4: `ImageFinding` builder, bilingual cutaneous lesion/ABCDE/anatomy text extractor, and `SAME_LESION` text-image linker."""

from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from datetime import date
from typing import Any

from medresearch_platform.extraction.contract import ImageFinding, RiskBand
from medresearch_platform.vision.cnn_multitask import (
    SAFETY_NOTICE_EN,
    SAFETY_NOTICE_FR,
    SkinPredictionResult,
)
from medresearch_platform.vision.router import TriageDecision

# Bilingual anatomical site dictionary -> (canonical_site, laterality)
_BODY_SITE_PATTERNS: tuple[tuple[re.Pattern[str], str, str | None], ...] = (
    (
        re.compile(
            r"\b(?:avant[-\s]bras\s+gauche|bras\s+gauche|épaule\s+gauche|poignet\s+gauche|main\s+gauche"
            r"|left\s+forearm|left\s+arm|left\s+upper\s+arm|left\s+shoulder|left\s+wrist|left\s+hand)\b",
            re.IGNORECASE,
        ),
        "upper extremity",
        "left",
    ),
    (
        re.compile(
            r"\b(?:avant[-\s]bras\s+droit|bras\s+droit|épaule\s+droite|poignet\s+droit|main\s+droite"
            r"|right\s+forearm|right\s+arm|right\s+upper\s+arm|right\s+shoulder|right\s+wrist|right\s+hand)\b",
            re.IGNORECASE,
        ),
        "upper extremity",
        "right",
    ),
    (
        re.compile(
            r"\b(?:avant[-\s]bras|membre\s+supérieur|bras|épaule|poignet|dos\s+de\s+la\s+main"
            r"|forearm|upper\s+extremity|upper\s+limb|arm|shoulder|wrist|back\s+of\s+hand)\b",
            re.IGNORECASE,
        ),
        "upper extremity",
        None,
    ),
    (
        re.compile(
            r"\b(?:jambe\s+gauche|cuisse\s+gauche|mollet\s+gauche|cheville\s+gauche|pied\s+gauche"
            r"|left\s+leg|left\s+thigh|left\s+calf|left\s+ankle|left\s+foot|left\s+lower\s+extremity)\b",
            re.IGNORECASE,
        ),
        "lower extremity",
        "left",
    ),
    (
        re.compile(
            r"\b(?:jambe\s+droite|cuisse\s+droite|mollet\s+droit|cheville\s+droite|pied\s+droit"
            r"|right\s+leg|right\s+thigh|right\s+calf|right\s+ankle|right\s+foot|right\s+lower\s+extremity)\b",
            re.IGNORECASE,
        ),
        "lower extremity",
        "right",
    ),
    (
        re.compile(
            r"\b(?:jambe|cuisse|mollet|membre\s+inférieur|cheville"
            r"|lower\s+extremity|lower\s+limb|leg|thigh|calf|ankle)\b",
            re.IGNORECASE,
        ),
        "lower extremity",
        None,
    ),
    (
        re.compile(
            r"\b(?:haut\s+du\s+dos|bas\s+du\s+dos|région\s+dorsale|région\s+scapulaire|tronc\s+postérieur|dos"
            r"|posterior\s+torso|upper\s+back|lower\s+back|scapular\s+region|torso\s+back|back)\b",
            re.IGNORECASE,
        ),
        "posterior torso",
        None,
    ),
    (
        re.compile(
            r"\b(?:thorax\s+antérieur|paroi\s+thoracique|poitrine|abdomen|tronc\s+antérieur|flanc"
            r"|anterior\s+torso|anterior\s+chest|chest\s+wall|chest|abdomen|torso\s+front|trunk)\b",
            re.IGNORECASE,
        ),
        "anterior torso",
        None,
    ),
    (
        re.compile(
            r"\b(?:visage|joue\s+gauche|joue\s+droite|joue|front|nez|oreille|cuir\s+chevelu|cou|tête"
            r"|head/neck|head\s+or\s+neck|face|cheek|forehead|scalp|neck|ear|nose)\b",
            re.IGNORECASE,
        ),
        "head/neck",
        None,
    ),
    (
        re.compile(
            r"\b(?:paume|plante\s+du\s+pied|région\s+palmo[-\s]plantaire"
            r"|palms/soles|palm|sole|foot\s+sole|palmoplantar)\b",
            re.IGNORECASE,
        ),
        "palms/soles",
        None,
    ),
)

_LESION_MENTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "SYMPTOM",
        re.compile(
            r"\b(?:lésion\s+pigmentée|lésion\s+cutanée|macule\s+pigmentée|macule\s+asymétrique"
            r"|grain\s+de\s+beauté|nodule\s+ulcéré|tache\s+pigmentée|papule\s+érythémateuse"
            r"|pigmented\s+lesion|skin\s+lesion|cutaneous\s+lesion|mole|asymmetric\s+macule"
            r"|ulcerated\s+nodule|dark\s+macule|erythematous\s+papule)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "DIAGNOSIS",
        re.compile(
            r"\b(?:mélanome(?:\s+suspecté)?|nævus(?:\s+mélanocytaire|\s+atypique)?|naevus"
            r"|carcinome\s+basocellulaire|carcinome\s+épidermoïde|kératose\s+actinique|kératose\s+séborrhéique"
            r"|dermatofibrome|angiome"
            r"|melanoma|melanocytic\s+nevus|atypical\s+nevus|dysplastic\s+nevus|nevus"
            r"|basal\s+cell\s+carcinoma|squamous\s+cell\s+carcinoma|actinic\s+keratosis|seborrheic\s+keratosis"
            r"|dermatofibroma|hemangioma)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "SYMPTOM",
        re.compile(
            r"\b(?:changement\s+récent|augmentation\s+de\s+taille|bords\s+irréguliers|polychromie|asymétrie"
            r"|recent\s+change|enlarging|irregular\s+borders|color\s+variegation|asymmetry"
            r"|\d+(?:[.,]\d+)?\s*(?:mm|cm)\s*(?:de\s+diamètre|in\s+diameter)?)\b",
            re.IGNORECASE,
        ),
    ),
)


def normalize_body_site(raw_site: str | None) -> tuple[str, str | None]:
    """Normalize any EN/FR body-site phrase into `(canonical_site, laterality)`."""
    if not raw_site or not str(raw_site).strip():
        return "other", None
    s = str(raw_site).strip()
    lat: str | None = None
    if re.search(r"\b(?:gauche|left)\b", s, re.IGNORECASE):
        lat = "left"
    elif re.search(r"\b(?:droit|droite|right)\b", s, re.IGNORECASE):
        lat = "right"

    for pat, canon, pat_lat in _BODY_SITE_PATTERNS:
        if pat.search(s):
            return canon, (lat or pat_lat)

    low = s.lower()
    for canon_name in (
        "upper extremity",
        "lower extremity",
        "posterior torso",
        "anterior torso",
        "head/neck",
        "palms/soles",
    ):
        if canon_name in low:
            return canon_name, lat
    return "other", lat


def extract_cutaneous_text_descriptors(
    text: str,
    *,
    existing_entities: Sequence[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    """
    Extract cutaneous `ANATOMY` body sites and lesion/ABCDE descriptors (size, colour,
    `changement récent`, ABCDE words, lesion mentions) not already covered by `existing_entities`.
    """
    occupied: list[tuple[int, int]] = []
    occupied_anatomy: list[tuple[int, int]] = []
    for e in existing_entities:
        cs = int(e.get("char_start") or e.get("start") or 0)
        ce = int(e.get("char_end") or e.get("end") or 0)
        if ce > cs:
            occupied.append((cs, ce))
            if str(e.get("entity_type") or e.get("type") or "") == "ANATOMY":
                occupied_anatomy.append((cs, ce))

    def _overlaps(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
        return any(not (end <= s or start >= e) for s, e in spans)

    added: list[dict[str, Any]] = []
    idx_counter = len(existing_entities) + 1

    # 1. Extract ANATOMY body site spans (only skip if an ANATOMY span already covers it)
    for pat, canon_site, _ in _BODY_SITE_PATTERNS:
        for m in pat.finditer(text):
            cs, ce = m.start(), m.end()
            if _overlaps(cs, ce, occupied_anatomy):
                continue
            occupied_anatomy.append((cs, ce))
            occupied.append((cs, ce))
            surf = text[cs:ce]
            added.append({
                "entity_id": f"ent_cut_anat_{idx_counter}",
                "entity_type": "ANATOMY",
                "source_text": surf,
                "normalized_text": canon_site,
                "assertion_status": "AFFIRMED",
                "temporal_status": "CURRENT",
                "char_start": cs,
                "char_end": ce,
                "confidence": 0.93,
                "review_status": "AI_PREDICTED",
                "source_stage": "cutaneous_descriptor_extractor",
            })
            idx_counter += 1

    # 2. Extract cutaneous lesion & ABCDE descriptor spans
    for ent_type, pat in _LESION_MENTION_PATTERNS:
        for m in pat.finditer(text):
            cs, ce = m.start(), m.end()
            if _overlaps(cs, ce, occupied):
                continue
            occupied.append((cs, ce))
            surf = text[cs:ce]
            added.append({
                "entity_id": f"ent_cut_les_{idx_counter}",
                "entity_type": ent_type,
                "source_text": surf,
                "normalized_text": surf.lower(),
                "assertion_status": "AFFIRMED",
                "temporal_status": "CURRENT",
                "char_start": cs,
                "char_end": ce,
                "confidence": 0.91,
                "review_status": "AI_PREDICTED",
                "source_stage": "cutaneous_descriptor_extractor",
            })
            idx_counter += 1

    added.sort(key=lambda d: int(d["char_start"]))
    return added


def _parse_iso_date(val: str | None) -> date | None:
    if not val:
        return None
    s = str(val).strip()[:10]
    try:
        return date.fromisoformat(s)
    except Exception:
        return None


def propose_same_lesion_links(
    text: str,
    entities: Sequence[dict[str, Any]],
    image_findings: Sequence[ImageFinding | dict[str, Any]],
    *,
    document_id: str,
    patient_id: str | None = None,
    document_date_iso: str | None = None,
) -> list[dict[str, Any]]:
    """
    Propose reviewable `SAME_LESION` links between text lesion/anatomy entities and
    `ImageFinding` records belonging to the same patient/encounter when body sites match.
    """
    # Ensure cutaneous descriptors are present
    extra_ents = extract_cutaneous_text_descriptors(text, existing_entities=entities)
    all_ents = list(entities) + extra_ents

    # Collect anatomy spans in text with their canonical body site and laterality
    anat_mentions: list[dict[str, Any]] = []
    lesion_mentions: list[dict[str, Any]] = []

    for e in all_ents:
        etype = str(e.get("entity_type") or e.get("type") or "")
        surf = str(e.get("source_text") or e.get("original_text") or e.get("text") or "")
        cs = int(e.get("char_start") or e.get("start") or 0)
        ce = int(e.get("char_end") or e.get("end") or 0)
        eid = str(e.get("entity_id") or e.get("id") or f"ent_{cs}_{ce}")
        assertion = str(e.get("assertion_status") or e.get("assertion") or "AFFIRMED").upper()
        if assertion in ("NEGATED", "ABSENT"):
            continue

        if etype == "ANATOMY":
            canon, lat = normalize_body_site(surf)
            if canon == "other":
                canon, lat = normalize_body_site(str(e.get("normalized_text") or ""))
            if canon != "other":
                anat_mentions.append({
                    "entity_id": eid,
                    "surface": surf,
                    "char_start": cs,
                    "char_end": ce,
                    "canonical_site": canon,
                    "laterality": lat,
                })
        elif etype in ("DIAGNOSIS", "SYMPTOM"):
            lesion_mentions.append({
                "entity_id": eid,
                "surface": surf,
                "char_start": cs,
                "char_end": ce,
            })

    doc_dt = _parse_iso_date(document_date_iso)
    links: list[dict[str, Any]] = []

    for finding_item in image_findings:
        f_dict = finding_item.to_dict() if isinstance(finding_item, ImageFinding) else dict(finding_item)
        img_id = str(f_dict.get("image_id") or "")
        img_site_raw = f_dict.get("body_site")
        img_canon, img_lat = normalize_body_site(img_site_raw)
        if img_canon == "other":
            continue

        img_dt = _parse_iso_date(f_dict.get("captured_date_iso"))
        date_delta: int | None = None
        if doc_dt is not None and img_dt is not None:
            date_delta = abs((doc_dt - img_dt).days)

        for anat in anat_mentions:
            if anat["canonical_site"] != img_canon:
                continue
            # Check laterality conflict (left vs right)
            if anat["laterality"] and img_lat and anat["laterality"] != img_lat:
                continue

            # Find nearest lesion mention within 140 chars in the same sentence/passage
            best_lesion = None
            best_dist = 999999
            for les in lesion_mentions:
                dist = min(
                    abs(int(les["char_start"]) - int(anat["char_end"])),
                    abs(int(anat["char_start"]) - int(les["char_end"])),
                )
                if dist <= 140 and dist < best_dist:
                    best_dist = dist
                    best_lesion = les

            if best_lesion is not None:
                lo = min(int(best_lesion["char_start"]), int(anat["char_start"]))
                hi = max(int(best_lesion["char_end"]), int(anat["char_end"]))
                target_eid = str(best_lesion["entity_id"])
                span_txt = text[lo:hi].replace("\n", " ")
            else:
                lo = int(anat["char_start"])
                hi = int(anat["char_end"])
                target_eid = str(anat["entity_id"])
                span_txt = str(anat["surface"])

            conf = 0.86
            if anat["laterality"] and img_lat and anat["laterality"] == img_lat:
                conf += 0.06
            if date_delta is not None:
                if date_delta <= 7:
                    conf += 0.03
                elif date_delta > 90:
                    conf -= 0.08
            conf = round(max(0.55, min(0.97, conf)), 4)

            links.append({
                "link_id": f"lnk_{uuid.uuid4().hex[:12]}",
                "relation_type": "SAME_LESION",
                "patient_id": patient_id or f_dict.get("patient_id"),
                "document_id": document_id,
                "image_id": img_id,
                "text_entity_id": target_eid,
                "text_span": span_txt,
                "char_start": lo,
                "char_end": hi,
                "matched_body_site": img_canon,
                "date_delta_days": date_delta,
                "confidence": conf,
                "review_status": "needs_review",
                "rationale_en": (
                    f"Matched cutaneous lesion mention '{span_txt}' at body site '{img_canon}' "
                    f"(confidence {conf:.2f}; human review required)."
                ),
                "rationale_fr": (
                    f"Correspondance anatomique entre la mention '{span_txt}' et la localisation '{img_canon}' "
                    f"(confiance {conf:.2f} ; validation humaine requise)."
                ),
            })

    return links


def build_image_finding(
    *,
    image_id: str,
    document_id: str | None,
    patient_id: str | None,
    triage: TriageDecision,
    skin_prediction: SkinPredictionResult | None,
    body_site: str | None = None,
    captured_date_iso: str | None = None,
    heatmap_uri: str | None = None,
    git_commit: str = "ce3140a",
    provenance: dict[str, Any] | None = None,
) -> ImageFinding:
    """Construct a contract-compliant `ImageFinding` from triage and skin model outputs."""
    canon_site, _ = normalize_body_site(body_site)
    resolved_site = canon_site if canon_site != "other" else body_site

    if skin_prediction is None or triage.route_action != "skin_vision_pipeline":
        return ImageFinding(
            image_id=image_id,
            document_id=document_id,
            patient_id=patient_id,
            picture_type=triage.predicted_class,
            quality_result=triage.quality.to_dict(),
            malignancy_risk_probability=None,
            malignancy_risk_band="abstained",
            top3_diagnoses=(),
            heatmap_uri=None,
            abstained=True,
            abstain_reason_en=triage.abstain_reason_en or "Image routed away from skin model.",
            abstain_reason_fr=triage.abstain_reason_fr or "Image non soumise au modèle cutané.",
            body_site=resolved_site,
            model_name=triage.model_name,
            model_version="1.0.0",
            git_commit=git_commit,
            review_status="needs_review",
            provenance=dict(provenance or {"router_confidence": triage.confidence}),
            captured_date_iso=captured_date_iso,
            safety_notice_en=SAFETY_NOTICE_EN,
            safety_notice_fr=SAFETY_NOTICE_FR,
        )

    band: RiskBand = "abstained"
    if skin_prediction.risk_band in ("low", "medium", "high", "abstained"):
        band = skin_prediction.risk_band  # type: ignore[assignment]

    return ImageFinding(
        image_id=image_id,
        document_id=document_id,
        patient_id=patient_id,
        picture_type=triage.predicted_class,
        quality_result=skin_prediction.quality.to_dict(),
        malignancy_risk_probability=(
            None if skin_prediction.abstained else skin_prediction.malignancy_probability
        ),
        malignancy_risk_band=band,
        top3_diagnoses=tuple(skin_prediction.top3_diagnoses),
        heatmap_uri=heatmap_uri,
        abstained=skin_prediction.abstained,
        abstain_reason_en=skin_prediction.abstain_reason_en,
        abstain_reason_fr=skin_prediction.abstain_reason_fr,
        body_site=resolved_site,
        model_name=skin_prediction.model_name,
        model_version="1.0.0",
        git_commit=git_commit,
        review_status="needs_review" if skin_prediction.needs_review else "unreviewed",
        provenance={
            **(provenance or {}),
            "router_class": triage.predicted_class,
            "router_confidence": triage.confidence,
            "energy_score": skin_prediction.energy_score,
            "predicted_modality": skin_prediction.predicted_modality,
        },
        captured_date_iso=captured_date_iso,
        safety_notice_en=SAFETY_NOTICE_EN,
        safety_notice_fr=SAFETY_NOTICE_FR,
    )
