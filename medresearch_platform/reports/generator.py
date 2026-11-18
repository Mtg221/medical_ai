"""Bilingual (EN/FR) traceable Markdown, HTML, and PDF report generator."""

from __future__ import annotations

import html
import io
from typing import Any, Literal, Mapping, Sequence

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from medresearch_platform.security.privacy import deidentify_clinical_text

ReportLanguage = Literal["en", "fr"]
ReportFormat = Literal["markdown", "html", "pdf"]


def _scrub_phi(val: Any, extra_pii: Sequence[Mapping[str, str]] | None = None) -> str:
    text = str(val)
    if extra_pii:
        for item in extra_pii:
            orig = item.get("original", "")
            tok = item.get("token", "[PHI_REDACTED]")
            if orig and orig in text:
                text = text.replace(orig, tok)
    return deidentify_clinical_text(text).scrubbed_text

_LABELS: dict[ReportLanguage, dict[str, str]] = {
    "en": {
        "title": "Structured Medical Information Extraction Report",
        "disclaimer": (
            "RESEARCH TOOL ONLY — NOT FOR CLINICAL DIAGNOSIS OR TREATMENT. "
            "Human clinician review is required before any clinical use."
        ),
        "metadata": "Document Metadata",
        "doc_id": "Document ID",
        "filename": "Filename",
        "language": "Detected Language",
        "mode": "Extraction Mode",
        "summary_counts": "Summary Counts",
        "total_facts": "Total Extracted Facts",
        "relations_title": "Extracted Relationships",
        "none_extracted": "None extracted from source.",
        "source_ref": "Source",
        "assertion": "Assertion",
        "confidence": "Confidence",
        "temporal": "Temporal",
        "negation_badge": "NEGATED / ABSENT",
        "uncertain_badge": "UNCERTAIN",
        "present_badge": "AFFIRMED",
        "categories": "Clinical Facts by Category",
        "to_verify_title": "To Verify (Low Confidence)",
        "skin_images_title": "Skin Images",
        "skin_safety": "Research tool — not a diagnosis. A dermatologist must examine the lesion.",
    },
    "fr": {
        "title": "Rapport structuré d'extraction d'information médicale",
        "disclaimer": (
            "OUTIL DE RECHERCHE UNIQUEMENT — NON DESTINÉ AU DIAGNOSTIC OU AU TRAITEMENT. "
            "Une révision humaine par un clinicien est requise avant tout usage clinique."
        ),
        "metadata": "Métadonnées du document",
        "doc_id": "Identifiant du document",
        "filename": "Nom du fichier",
        "language": "Langue détectée",
        "mode": "Mode d'extraction",
        "summary_counts": "Résumé quantitatif",
        "total_facts": "Nombre total de faits extraits",
        "relations_title": "Relations extraites",
        "none_extracted": "Aucun élément extrait de la source.",
        "source_ref": "Source",
        "assertion": "Statut d'assertion",
        "confidence": "Confiance",
        "temporal": "Temporalité",
        "negation_badge": "NÉGATION / ABSENT",
        "uncertain_badge": "INCERTAIN",
        "present_badge": "AFFIRMÉ",
        "categories": "Faits cliniques par catégorie",
        "to_verify_title": "À vérifier (faible confiance)",
        "skin_images_title": "Images cutanées",
        "skin_safety": "Outil de recherche — ne constitue pas un diagnostic. Un dermatologue doit examiner la lésion.",
    },
}

_CATEGORY_GROUPS: tuple[tuple[str, dict[ReportLanguage, str], tuple[str, ...]], ...] = (
    ("DIAGNOSIS", {"en": "Diagnoses & Conditions", "fr": "Diagnostics & Pathologies"}, ("DIAGNOSIS", "PROBLEM", "CONDITION", "DISEASE")),
    ("SYMPTOM", {"en": "Symptoms & Findings", "fr": "Symptômes & Signes cliniques"}, ("SYMPTOM", "SIGN", "FINDING")),
    ("MEDICATION", {"en": "Medications, Dosages & Routes", "fr": "Traitements, Posologies & Voies"}, ("MEDICATION", "DRUG", "DOSAGE", "FREQUENCY", "ROUTE")),
    ("LAB", {"en": "Laboratory Tests & Values", "fr": "Examens biologiques & Valeurs"}, ("LAB_TEST", "LAB_VALUE", "LAB_UNIT", "MEASUREMENT")),
    ("PROCEDURE", {"en": "Procedures & Imaging", "fr": "Actes, Procédures & Imagerie"}, ("PROCEDURE", "IMAGING", "TEST")),
    ("TEMPORAL", {"en": "Dates, Durations & Temporal Expressions", "fr": "Dates, Durées & Expressions temporelles"}, ("TIMEX", "DATE", "DURATION", "TEMPORAL")),
)


def _assertion_label(assertion: str, lang: ReportLanguage) -> str:
    lbl = _LABELS[lang]
    norm = assertion.lower()
    if norm in ("absent", "negated"):
        return f"{lbl['negation_badge']} ({assertion})"
    if norm in ("uncertain", "hypothetical"):
        return f"{lbl['uncertain_badge']} ({assertion})"
    return f"{lbl['present_badge']} ({assertion})"


def _group_entities(
    entities: Sequence[Mapping[str, Any]],
    lang: ReportLanguage,
) -> list[tuple[str, list[Mapping[str, Any]]]]:
    assigned_indices: set[int] = set()
    grouped: list[tuple[str, list[Mapping[str, Any]]]] = []
    for _, titles, types in _CATEGORY_GROUPS:
        bucket: list[Mapping[str, Any]] = []
        for idx, ent in enumerate(entities):
            etype = str(ent.get("entity_type", "")).upper()
            if etype in types:
                bucket.append(ent)
                assigned_indices.add(idx)
        grouped.append((titles[lang], bucket))

    other = [ent for idx, ent in enumerate(entities) if idx not in assigned_indices]
    if other:
        other_title = "Other Clinical Mentions" if lang == "en" else "Autres mentions cliniques"
        grouped.append((other_title, other))
    return grouped


def render_markdown_report(
    document: Mapping[str, Any],
    entities: Sequence[Mapping[str, Any]],
    relations: Sequence[Mapping[str, Any]],
    *,
    image_findings: Sequence[Mapping[str, Any]] = (),
    same_lesion_links: Sequence[Mapping[str, Any]] = (),
    lang: ReportLanguage = "en",
) -> str:
    """Render a traceable Markdown report in English or French."""
    lbl = _LABELS[lang]
    extra_pii = document.get("pii_spans")
    doc_id = str(document.get("document_id", "unknown"))
    filename = _scrub_phi(document.get("original_filename", "document"), extra_pii)
    det_lang = str(document.get("detected_language", "unknown"))
    mode = str(document.get("extraction_mode", "NATIVE_TEXT"))

    confirmed = [e for e in entities if e.get("review_status") != "needs_review"]
    to_verify = [e for e in entities if e.get("review_status") == "needs_review"]

    lines: list[str] = [
        f"# {lbl['title']}",
        "",
        f"> **WARNING:** {lbl['disclaimer']}",
        "",
        f"## {lbl['metadata']}",
        f"- **{lbl['doc_id']}:** `{doc_id}`",
        f"- **{lbl['filename']}:** `{filename}`",
        f"- **{lbl['language']}:** `{det_lang}`",
        f"- **{lbl['mode']}:** `{mode}`",
        "",
        f"## {lbl['summary_counts']}",
        f"- **{lbl['total_facts']}:** {len(confirmed)}",
        "",
        f"## {lbl['categories']}",
        "",
    ]

    for cat_title, items in _group_entities(confirmed, lang):
        lines.append(f"### {cat_title}")
        if not items:
            lines.append(f"- *{lbl['none_extracted']}*")
            lines.append("")
            continue
        for ent in items:
            surf = _scrub_phi(ent.get("surface_text") or ent.get("source_text") or ent.get("extracted_value") or "", extra_pii)
            norm = _scrub_phi(ent.get("normalized_text") or ent.get("normalized_concept") or surf, extra_pii)
            etype = str(ent.get("entity_type", "UNKNOWN"))
            assertion = str(ent.get("assertion_status", "present"))
            conf = float(ent.get("confidence", ent.get("confidence_score", 0.0)) or 0.0)
            page = ent.get("page_number", 1)
            c0 = ent.get("char_start", "?")
            c1 = ent.get("char_end", "?")
            temp = ent.get("temporal_status") or ent.get("temporal_information")
            temp_part = f" | {lbl['temporal']}: `{temp}`" if temp else ""
            lines.append(
                f"- **{surf}** (`{etype}` -> `{norm}`) | "
                f"{lbl['assertion']}: **{_assertion_label(assertion, lang)}** | "
                f"{lbl['confidence']}: `{conf:.2f}`{temp_part} | "
                f"{lbl['source_ref']}: `[doc={doc_id}, p.{page}, chars {c0}-{c1}]`"
            )
        lines.append("")

    if to_verify:
        lines.append(f"## {lbl['to_verify_title']}")
        for ent in to_verify:
            surf = _scrub_phi(ent.get("surface_text") or ent.get("source_text") or ent.get("extracted_value") or "", extra_pii)
            norm = _scrub_phi(ent.get("normalized_text") or ent.get("normalized_concept") or surf, extra_pii)
            etype = str(ent.get("entity_type", "UNKNOWN"))
            assertion = str(ent.get("assertion_status", "present"))
            conf = float(ent.get("confidence", ent.get("confidence_score", 0.0)) or 0.0)
            page = ent.get("page_number", 1)
            c0 = ent.get("char_start", "?")
            c1 = ent.get("char_end", "?")
            lines.append(
                f"- **{surf}** (`{etype}` -> `{norm}`) | "
                f"{lbl['assertion']}: **{_assertion_label(assertion, lang)}** | "
                f"{lbl['confidence']}: `{conf:.2f}` | "
                f"{lbl['source_ref']}: `[doc={doc_id}, p.{page}, chars {c0}-{c1}]`"
            )
        lines.append("")

    lines.append(f"## {lbl['relations_title']}")
    if not relations:
        lines.append(f"- *{lbl['none_extracted']}*")
    else:
        ent_by_id = {str(e.get("entity_id", "")): e for e in entities if e.get("entity_id")}
        for rel in relations:
            src = ent_by_id.get(str(rel.get("source_entity_id", "")), {})
            tgt = ent_by_id.get(str(rel.get("target_entity_id", "")), {})
            src_txt = _scrub_phi(src.get("surface_text") or src.get("source_text") or rel.get("source_entity_id") or "?", extra_pii)
            tgt_txt = _scrub_phi(tgt.get("surface_text") or tgt.get("source_text") or rel.get("target_value") or rel.get("target_entity_id") or "?", extra_pii)
            rtype = str(rel.get("relation_type", "RELATED_TO"))
            conf = float(rel.get("confidence", 0.0) or 0.0)
            lines.append(f"- `{src_txt}` — **[{rtype}]** → `{tgt_txt}` ({lbl['confidence']}: `{conf:.2f}`)")
    lines.append("")

    if image_findings:
        lines.append(f"## {lbl['skin_images_title']}")
        lines.append(f"> {lbl['skin_safety']}")
        lines.append(
            "> Carte de chaleur Grad-CAM omise du rapport (échec du test de cohérence : décalage Adebayo 0,227 < 0,30)."
            if lang == "fr"
            else "> Grad-CAM heat map omitted from report (failed randomization sanity check: Adebayo shift 0.227 < 0.30)."
        )
        lines.append("")
        for f in image_findings:
            img_id = str(f.get("image_id", "unknown"))
            ptype = str(f.get("picture_type", "skin_dermoscopic"))
            band = str(f.get("malignancy_risk_band", "abstained"))
            prob = f.get("malignancy_risk_probability")
            rev = str(f.get("review_status", "needs_review"))
            site = str(f.get("body_site") or ("unspecified" if lang == "en" else "non spécifié"))
            abstained = bool(f.get("abstained", False))
            top3 = f.get("top3_diagnoses") or []
            if isinstance(top3, str):
                import json as _json
                try:
                    top3 = _json.loads(top3)
                except Exception:
                    top3 = []
            top3_str = ", ".join(f"{d.get('diagnosis')} ({float(d.get('probability', 0.0)):.2f})" for d in top3[:3])
            if abstained or prob is None:
                reason = f.get("abstain_reason_fr" if lang == "fr" else "abstain_reason_en") or "Abstained"
                lines.append(
                    f"- **`{img_id}`** (`{ptype}`, site: `{site}`) | "
                    f"**ABSTAINED**: {reason} | Review: `{rev}`"
                )
            elif ptype == "skin_clinical":
                clin_msg = (
                    "Aucune catégorie de risque affichée (photo par téléphone : sensibilité servie 21,6 % < 50 % — veuillez consulter un dermatologue)"
                    if lang == "fr"
                    else "No risk band shown (phone photo: served sensitivity 21.6% < 50% — please see a dermatologist)"
                )
                lines.append(
                    f"- **`{img_id}`** (`{ptype}`, site: `{site}`) | "
                    f"**{clin_msg}** | Top-3: `{top3_str}` | Review: `{rev}`"
                )
            else:
                lines.append(
                    f"- **`{img_id}`** (`{ptype}`, site: `{site}`) | "
                    f"Risk band: **`{band.upper()}`** (`{float(prob):.2f}`) | "
                    f"Top-3: `{top3_str}` | Review: `{rev}`"
                )
        for lnk in same_lesion_links:
            lines.append(
                f"  - `SAME_LESION` link: `{lnk.get('image_id')}` ↔ "
                f"*{lnk.get('text_span')}* (`{lnk.get('matched_body_site')}`, conf `{float(lnk.get('confidence', 0.0)):.2f}`)"
            )
        lines.append("")

    return "\n".join(lines)


def render_html_report(
    document: Mapping[str, Any],
    entities: Sequence[Mapping[str, Any]],
    relations: Sequence[Mapping[str, Any]],
    *,
    image_findings: Sequence[Mapping[str, Any]] = (),
    same_lesion_links: Sequence[Mapping[str, Any]] = (),
    lang: ReportLanguage = "en",
) -> str:
    """Render a self-contained, accessible HTML clinical extraction report."""
    lbl = _LABELS[lang]
    extra_pii = document.get("pii_spans")
    doc_id = html.escape(str(document.get("document_id", "unknown")))
    filename = html.escape(_scrub_phi(document.get("original_filename", "document"), extra_pii))
    det_lang = html.escape(str(document.get("detected_language", "unknown")))
    mode = html.escape(str(document.get("extraction_mode", "NATIVE_TEXT")))

    confirmed = [e for e in entities if e.get("review_status") != "needs_review"]
    to_verify = [e for e in entities if e.get("review_status") == "needs_review"]

    sections_html: list[str] = []
    for cat_title, items in _group_entities(confirmed, lang):
        rows_html: list[str] = []
        if not items:
            rows_html.append(f"<p class='empty'>{html.escape(lbl['none_extracted'])}</p>")
        else:
            rows_html.append("<ul>")
            for ent in items:
                surf = html.escape(_scrub_phi(ent.get("surface_text") or ent.get("source_text") or ent.get("extracted_value") or "", extra_pii))
                norm = html.escape(_scrub_phi(ent.get("normalized_text") or ent.get("normalized_concept") or surf, extra_pii))
                etype = html.escape(str(ent.get("entity_type", "UNKNOWN")))
                assertion = str(ent.get("assertion_status", "present"))
                conf = float(ent.get("confidence", ent.get("confidence_score", 0.0)) or 0.0)
                page = html.escape(str(ent.get("page_number", 1)))
                c0 = html.escape(str(ent.get("char_start", "?")))
                c1 = html.escape(str(ent.get("char_end", "?")))
                badge_cls = "neg" if assertion.lower() in ("absent", "negated") else "pos"
                rows_html.append(
                    f"<li><strong>{surf}</strong> (<code>{etype}</code> &rarr; <code>{norm}</code>) "
                    f"<span class='badge {badge_cls}'>{html.escape(_assertion_label(assertion, lang))}</span> "
                    f"<span class='src'>[{html.escape(lbl['source_ref'])}: doc={doc_id}, p.{page}, chars {c0}-{c1}, conf={conf:.2f}]</span></li>"
                )
            rows_html.append("</ul>")
        sections_html.append(f"<section><h3>{html.escape(cat_title)}</h3>{''.join(rows_html)}</section>")

    if to_verify:
        tv_rows: list[str] = ["<ul>"]
        for ent in to_verify:
            surf = html.escape(_scrub_phi(ent.get("surface_text") or ent.get("source_text") or ent.get("extracted_value") or "", extra_pii))
            norm = html.escape(_scrub_phi(ent.get("normalized_text") or ent.get("normalized_concept") or surf, extra_pii))
            etype = html.escape(str(ent.get("entity_type", "UNKNOWN")))
            assertion = str(ent.get("assertion_status", "present"))
            conf = float(ent.get("confidence", ent.get("confidence_score", 0.0)) or 0.0)
            page = html.escape(str(ent.get("page_number", 1)))
            c0 = html.escape(str(ent.get("char_start", "?")))
            c1 = html.escape(str(ent.get("char_end", "?")))
            tv_rows.append(
                f"<li><strong>{surf}</strong> (<code>{etype}</code> &rarr; <code>{norm}</code>) "
                f"<span class='badge neg'>{html.escape(_assertion_label(assertion, lang))}</span> "
                f"<span class='src'>[{html.escape(lbl['source_ref'])}: doc={doc_id}, p.{page}, chars {c0}-{c1}, conf={conf:.2f}]</span></li>"
            )
        tv_rows.append("</ul>")
        sections_html.append(f"<section><h3>{html.escape(lbl['to_verify_title'])}</h3>{''.join(tv_rows)}</section>")

    if image_findings:
        heatmap_note = (
            "Carte de chaleur Grad-CAM omise du rapport (échec du test de cohérence : décalage Adebayo 0,227 < 0,30)."
            if lang == "fr"
            else "Grad-CAM heat map omitted from report (failed randomization sanity check: Adebayo shift 0.227 < 0.30)."
        )
        img_rows: list[str] = [
            f"<p class='disclaimer'>{html.escape(lbl['skin_safety'])}</p>",
            f"<p class='disclaimer'>{html.escape(heatmap_note)}</p><ul>",
        ]
        for f in image_findings:
            img_id = html.escape(str(f.get("image_id", "unknown")))
            ptype = html.escape(str(f.get("picture_type", "skin_dermoscopic")))
            band = html.escape(str(f.get("malignancy_risk_band", "abstained")))
            prob = f.get("malignancy_risk_probability")
            rev = html.escape(str(f.get("review_status", "needs_review")))
            site = html.escape(str(f.get("body_site") or ("unspecified" if lang == "en" else "non spécifié")))
            abstained = bool(f.get("abstained", False))
            top3 = f.get("top3_diagnoses") or []
            if isinstance(top3, str):
                import json as _json
                try:
                    top3 = _json.loads(top3)
                except Exception:
                    top3 = []
            top3_str = html.escape(", ".join(f"{d.get('diagnosis')} ({float(d.get('probability', 0.0)):.2f})" for d in top3[:3]))
            if abstained or prob is None:
                reason = html.escape(str(f.get("abstain_reason_fr" if lang == "fr" else "abstain_reason_en") or "Abstained"))
                img_rows.append(
                    f"<li><strong>{img_id}</strong> (<code>{ptype}</code>, site: <code>{site}</code>) "
                    f"<span class='badge neg'>ABSTAINED</span> {reason} [review: <code>{rev}</code>]</li>"
                )
            elif ptype == "skin_clinical":
                clin_msg = html.escape(
                    "Aucune catégorie de risque affichée (photo par téléphone : sensibilité servie 21,6 % < 50 % — veuillez consulter un dermatologue)"
                    if lang == "fr"
                    else "No risk band shown (phone photo: served sensitivity 21.6% < 50% — please see a dermatologist)"
                )
                img_rows.append(
                    f"<li><strong>{img_id}</strong> (<code>{ptype}</code>, site: <code>{site}</code>) "
                    f"<span class='badge neg'>{clin_msg}</span> "
                    f"Top-3: <code>{top3_str}</code> [review: <code>{rev}</code>]</li>"
                )
            else:
                img_rows.append(
                    f"<li><strong>{img_id}</strong> (<code>{ptype}</code>, site: <code>{site}</code>) "
                    f"<span class='badge pos'>{band.upper()} ({float(prob):.2f})</span> "
                    f"Top-3: <code>{top3_str}</code> [review: <code>{rev}</code>]</li>"
                )
        img_rows.append("</ul>")
        sections_html.append(f"<section><h3>{html.escape(lbl['skin_images_title'])}</h3>{''.join(img_rows)}</section>")

    return f"""<!doctype html>
<html lang="{lang}">
<head>
<meta charset="utf-8" />
<title>{html.escape(lbl['title'])}</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #0f172a; }}
  .disclaimer {{ background: #fef3c7; border: 1px solid #f59e0b; padding: 0.75rem 1rem; border-radius: 6px; font-weight: 600; font-size: 0.875rem; }}
  .meta {{ background: #f8fafc; border: 1px solid #e2e8f0; padding: 0.75rem 1rem; border-radius: 6px; margin: 1rem 0; font-size: 0.875rem; }}
  .badge {{ padding: 0.15rem 0.45rem; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }}
  .badge.pos {{ background: #dcfce7; color: #166534; }}
  .badge.neg {{ background: #fee2e2; color: #991b1b; }}
  .src {{ color: #475569; font-size: 0.8rem; }}
  .empty {{ color: #64748b; font-style: italic; font-size: 0.875rem; }}
</style>
</head>
<body>
  <div class="disclaimer" role="status">{html.escape(lbl['disclaimer'])}</div>
  <h1>{html.escape(lbl['title'])}</h1>
  <div class="meta">
    <strong>{html.escape(lbl['doc_id'])}:</strong> {doc_id} |
    <strong>{html.escape(lbl['filename'])}:</strong> {filename} |
    <strong>{html.escape(lbl['language'])}:</strong> {det_lang} |
    <strong>{html.escape(lbl['mode'])}:</strong> {mode} |
    <strong>{html.escape(lbl['total_facts'])}:</strong> {len(confirmed)}
  </div>
  <h2>{html.escape(lbl['categories'])}</h2>
  {''.join(sections_html)}
</body>
</html>
"""


def render_pdf_report(
    document: Mapping[str, Any],
    entities: Sequence[Mapping[str, Any]],
    relations: Sequence[Mapping[str, Any]],
    *,
    image_findings: Sequence[Mapping[str, Any]] = (),
    same_lesion_links: Sequence[Mapping[str, Any]] = (),
    lang: ReportLanguage = "en",
) -> bytes:
    """Render a print-ready PDF clinical extraction report via ReportLab."""
    md = render_markdown_report(
        document,
        entities,
        relations,
        image_findings=image_findings,
        same_lesion_links=same_lesion_links,
        lang=lang,
    )
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    story: list[Any] = []
    for raw_line in md.splitlines():
        line = raw_line.strip()
        if not line:
            story.append(Spacer(1, 6))
            continue
        escaped = html.escape(line.lstrip("#>- ").strip())
        if line.startswith("# "):
            story.append(Paragraph(escaped, styles["Heading1"]))
        elif line.startswith("## "):
            story.append(Paragraph(escaped, styles["Heading2"]))
        elif line.startswith("### "):
            story.append(Paragraph(escaped, styles["Heading3"]))
        else:
            story.append(Paragraph(escaped, styles["BodyText"]))
    doc.build(story)
    return buf.getvalue()


def generate_report_bytes(
    document: Mapping[str, Any],
    entities: Sequence[Mapping[str, Any]],
    relations: Sequence[Mapping[str, Any]],
    *,
    image_findings: Sequence[Mapping[str, Any]] = (),
    same_lesion_links: Sequence[Mapping[str, Any]] = (),
    lang: ReportLanguage = "en",
    fmt: ReportFormat = "markdown",
) -> tuple[bytes, str]:
    """Generate report bytes and MIME media type for Markdown, HTML, or PDF."""
    if fmt == "markdown":
        return (
            render_markdown_report(
                document,
                entities,
                relations,
                image_findings=image_findings,
                same_lesion_links=same_lesion_links,
                lang=lang,
            ).encode("utf-8"),
            "text/markdown; charset=utf-8",
        )
    if fmt == "html":
        return (
            render_html_report(
                document,
                entities,
                relations,
                image_findings=image_findings,
                same_lesion_links=same_lesion_links,
                lang=lang,
            ).encode("utf-8"),
            "text/html; charset=utf-8",
        )
    if fmt == "pdf":
        return (
            render_pdf_report(
                document,
                entities,
                relations,
                image_findings=image_findings,
                same_lesion_links=same_lesion_links,
                lang=lang,
            ),
            "application/pdf",
        )
    raise ValueError(f"Unsupported report format: {fmt}")
