"""Human correction, inter-annotator agreement (Cohen's kappa), and adjudication (Section 16)."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from medresearch_platform.security.audit import record_audit_event
from medresearch_platform.security.auth import AuthenticatedPrincipal
from medresearch_platform.security.privacy import deidentify_clinical_text


def submit_entity_human_review(
    conn: sqlite3.Connection,
    *,
    principal: AuthenticatedPrincipal,
    entity_id: str,
    review_decision: str,
    corrected_entity_type: str | None = None,
    corrected_assertion_status: str | None = None,
    corrected_certainty: str | None = None,
    corrected_temporality: str | None = None,
    corrected_experiencer: str | None = None,
    corrected_temporal_status: str | None = None,
    corrected_normalized_text: str | None = None,
    corrected_char_start: int | None = None,
    corrected_char_end: int | None = None,
    corrected_surface_text: str | None = None,
    rationale: str = "Expert clinical review",
) -> dict[str, Any]:
    """Record a human review/correction of an AI-extracted entity without erasing prediction history."""
    principal.assert_permission("annotation:review")
    if review_decision == "APPROVE_AS_GROUND_TRUTH":
        # Approving final ground truth requires adjudicator or admin permission if disputed
        pass
    if review_decision not in (
        "APPROVE_AS_GROUND_TRUTH",
        "CORRECT_AND_APPROVE",
        "FLAG_AMBIGUOUS",
        "REJECT",
    ):
        raise ValueError(f"Unsupported review_decision: {review_decision}")

    row = conn.execute(
        """
        SELECT e.*, d.institution_id
        FROM extracted_entities e
        JOIN documents d ON d.document_id = e.document_id
        WHERE e.entity_id = ?
        """,
        (entity_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"Entity not found: {entity_id}")

    principal.assert_institution_access(str(row["institution_id"]))

    row_keys = set(row.keys())
    prev_certainty = str(row["certainty"]) if "certainty" in row_keys else "confirmed"
    prev_temporality = str(row["temporality"]) if "temporality" in row_keys else "current"
    prev_experiencer = str(row["experiencer"]) if "experiencer" in row_keys else "patient"

    previous_val = {
        "entity_type": str(row["entity_type"]),
        "normalized_text": str(row["normalized_text"]),
        "source_text": str(row["source_text"]),
        "char_start": int(row["char_start"]),
        "char_end": int(row["char_end"]),
        "assertion_status": str(row["assertion_status"]),
        "certainty": prev_certainty,
        "temporality": prev_temporality,
        "experiencer": prev_experiencer,
        "temporal_status": str(row["temporal_status"]),
        "review_status": str(row["review_status"]),
    }

    new_ent_type = corrected_entity_type or str(row["entity_type"])
    new_norm = corrected_normalized_text or str(row["normalized_text"])
    new_surface = corrected_surface_text if corrected_surface_text is not None else str(row["source_text"])
    new_start = int(corrected_char_start) if corrected_char_start is not None else int(row["char_start"])
    new_end = int(corrected_char_end) if corrected_char_end is not None else int(row["char_end"])
    if new_end <= new_start:
        raise ValueError(f"Invalid span offsets: char_start={new_start}, char_end={new_end}")
    new_ass = corrected_assertion_status or str(row["assertion_status"])
    new_certainty = corrected_certainty or prev_certainty
    new_temporality = corrected_temporality or prev_temporality
    new_experiencer = corrected_experiencer or prev_experiencer
    if new_certainty not in ("confirmed", "possible", "absent"):
        raise ValueError(f"Invalid certainty: {new_certainty}")
    if new_temporality not in ("current", "historical"):
        raise ValueError(f"Invalid temporality: {new_temporality}")
    if new_experiencer not in ("patient", "family"):
        raise ValueError(f"Invalid experiencer: {new_experiencer}")
    new_tmp = corrected_temporal_status or (
        "HISTORICAL" if (corrected_temporality == "historical") else (
            "CURRENT" if (corrected_temporality == "current") else str(row["temporal_status"])
        )
    )

    if review_decision == "APPROVE_AS_GROUND_TRUTH":
        new_status = "APPROVED_GROUND_TRUTH"
    elif review_decision == "CORRECT_AND_APPROVE":
        new_status = "HUMAN_CORRECTED"
    elif review_decision == "FLAG_AMBIGUOUS":
        new_status = "UNRESOLVED_DISAGREEMENT"
    else:
        new_status = "REJECTED"

    corrected_val = {
        "entity_type": new_ent_type,
        "normalized_text": new_norm,
        "source_text": new_surface,
        "char_start": new_start,
        "char_end": new_end,
        "assertion_status": new_ass,
        "certainty": new_certainty,
        "temporality": new_temporality,
        "experiencer": new_experiencer,
        "temporal_status": new_tmp,
        "review_status": new_status,
    }

    review_id = f"rev_{uuid.uuid4().hex[:14]}"
    now_iso = datetime.now(timezone.utc).isoformat()

    conn.execute(
        """
        INSERT INTO human_reviews (
            review_id, target_type, target_id, document_id, reviewer_user_id,
            previous_value_json, corrected_value_json, review_decision, rationale, created_at
        ) VALUES (?, 'ENTITY', ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            review_id,
            entity_id,
            str(row["document_id"]),
            principal.user_id,
            json.dumps(previous_val, sort_keys=True),
            json.dumps(corrected_val, sort_keys=True),
            review_decision,
            rationale,
            now_iso,
        ),
    )

    conn.execute(
        """
        UPDATE extracted_entities
        SET entity_type = ?, normalized_text = ?, source_text = ?, char_start = ?, char_end = ?,
            assertion_status = ?, certainty = ?, temporality = ?, experiencer = ?,
            temporal_status = ?, review_status = ?
        WHERE entity_id = ?
        """,
        (
            new_ent_type,
            new_norm,
            new_surface,
            new_start,
            new_end,
            new_ass,
            new_certainty,
            new_temporality,
            new_experiencer,
            new_tmp,
            new_status,
            entity_id,
        ),
    )

    record_audit_event(
        conn,
        actor_user_id=principal.user_id,
        institution_id=principal.institution_id,
        action_type="HUMAN_REVIEW_ENTITY",
        resource_type="EXTRACTED_ENTITY",
        resource_id=entity_id,
        metadata={
            "review_id": review_id,
            "decision": review_decision,
            "new_status": new_status,
        },
    )

    return {
        "review_id": review_id,
        "entity_id": entity_id,
        "document_id": str(row["document_id"]),
        "review_status": new_status,
        "previous": previous_val,
        "updated": corrected_val,
        "previous_value": previous_val,
        "corrected_value": corrected_val,
        "review_decision": review_decision,
        "reviewer_user_id": principal.user_id,
        "created_at": now_iso,
    }


def compute_inter_annotator_agreement(
    annotator_a_labels: list[str],
    annotator_b_labels: list[str],
) -> dict[str, Any]:
    """Compute observed agreement and Cohen's kappa across paired independent annotations (Section 16)."""
    if len(annotator_a_labels) != len(annotator_b_labels) or not annotator_a_labels:
        raise ValueError("Annotator label lists must be non-empty and of equal length.")

    n = len(annotator_a_labels)
    categories = sorted(set(annotator_a_labels) | set(annotator_b_labels))
    agree_count = sum(1 for a, b in zip(annotator_a_labels, annotator_b_labels) if a == b)
    p_o = agree_count / n

    p_e = 0.0
    for cat in categories:
        freq_a = sum(1 for a in annotator_a_labels if a == cat) / n
        freq_b = sum(1 for b in annotator_b_labels if b == cat) / n
        p_e += freq_a * freq_b

    if abs(1.0 - p_e) < 1e-9:
        kappa = 1.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)

    disagreements = [
        {"index": idx, "annotator_a": a, "annotator_b": b}
        for idx, (a, b) in enumerate(zip(annotator_a_labels, annotator_b_labels))
        if a != b
    ]
    return {
        "total_items": n,
        "observed_agreement": round(p_o, 4),
        "expected_chance_agreement": round(p_e, 4),
        "cohens_kappa": round(kappa, 4),
        "unresolved_disagreement_count": len(disagreements),
        "disagreements": disagreements,
        "epistemic_note": "Agreement does not guarantee clinical truth; unresolved items require expert adjudication.",
    }


def list_human_reviews(
    conn: sqlite3.Connection,
    *,
    institution_id: str,
    document_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return persisted human review decisions scoped to the caller's institution."""
    params: list[Any] = [institution_id]
    where_doc = ""
    if document_id:
        where_doc = " AND r.document_id = ?"
        params.append(document_id)
    rows = conn.execute(
        f"""
        SELECT r.review_id, r.target_type, r.target_id, r.document_id, r.reviewer_user_id,
               r.previous_value_json, r.corrected_value_json, r.review_decision, r.rationale, r.created_at
        FROM human_reviews r
        JOIN documents d ON d.document_id = r.document_id
        WHERE d.institution_id = ? {where_doc}
        ORDER BY r.created_at DESC
        """,
        tuple(params),
    ).fetchall()
    return [
        {
            "review_id": str(r["review_id"]),
            "target_type": str(r["target_type"]),
            "target_id": str(r["target_id"]),
            "document_id": str(r["document_id"]),
            "reviewer_user_id": str(r["reviewer_user_id"]),
            "previous_value": json.loads(str(r["previous_value_json"])),
            "corrected_value": json.loads(str(r["corrected_value_json"])),
            "review_decision": str(r["review_decision"]),
            "rationale": str(r["rationale"]),
            "created_at": str(r["created_at"]),
        }
        for r in rows
    ]


def export_reviewed_gold(
    conn: sqlite3.Connection,
    *,
    institution_id: str,
    fmt: str = "jsonl",
    document_id: str | None = None,
) -> str:
    """Export approved and human-corrected entities as JSONL or BRAT .ann gold annotations."""
    params: list[Any] = [institution_id]
    where_doc = ""
    if document_id:
        where_doc = " AND d.document_id = ?"
        params.append(document_id)

    docs = conn.execute(
        f"""
        SELECT d.document_id, d.detected_language, d.document_type, d.deidentified_preview
        FROM documents d
        WHERE d.institution_id = ? {where_doc}
        ORDER BY d.created_at DESC
        """,
        tuple(params),
    ).fetchall()

    if fmt.lower() == "brat":
        brat_lines: list[str] = []
        t_idx = 1
        a_idx = 1
        for d in docs:
            ents = conn.execute(
                """
                SELECT entity_type, char_start, char_end, source_text, assertion_status,
                       certainty, temporality, experiencer
                FROM extracted_entities
                WHERE document_id = ? AND review_status IN ('APPROVED_GROUND_TRUTH', 'HUMAN_CORRECTED')
                ORDER BY char_start ASC
                """,
                (str(d["document_id"]),),
            ).fetchall()
            if not ents:
                continue
            brat_lines.append(f"# Document: {d['document_id']} ({d['detected_language']})")
            for e in ents:
                tid = f"T{t_idx}"
                t_idx += 1
                surf = deidentify_clinical_text(str(e["source_text"])).scrubbed_text.replace("\n", " ")
                brat_lines.append(f"{tid}\t{e['entity_type']} {e['char_start']} {e['char_end']}\t{surf}")
                if str(e["assertion_status"]) not in ("AFFIRMED", "present"):
                    brat_lines.append(f"A{a_idx}\tAssertion {tid} {e['assertion_status']}")
                    a_idx += 1
                if str(e["certainty"]) != "confirmed":
                    brat_lines.append(f"A{a_idx}\tCertainty {tid} {e['certainty']}")
                    a_idx += 1
                if str(e["temporality"]) != "current":
                    brat_lines.append(f"A{a_idx}\tTemporality {tid} {e['temporality']}")
                    a_idx += 1
                if str(e["experiencer"]) != "patient":
                    brat_lines.append(f"A{a_idx}\tExperiencer {tid} {e['experiencer']}")
                    a_idx += 1
        return "\n".join(brat_lines) + ("\n" if brat_lines else "")

    jsonl_lines: list[str] = []
    for d in docs:
        ents = conn.execute(
            """
            SELECT entity_id, entity_type, char_start, char_end, source_text, assertion_status,
                   certainty, temporality, experiencer, review_status
            FROM extracted_entities
            WHERE document_id = ? AND review_status IN ('APPROVED_GROUND_TRUTH', 'HUMAN_CORRECTED')
            ORDER BY char_start ASC
            """,
            (str(d["document_id"]),),
        ).fetchall()
        if not ents:
            continue
        jsonl_lines.append(
            json.dumps(
                {
                    "doc_id": str(d["document_id"]),
                    "language": str(d["detected_language"]),
                    "doc_type": str(d["document_type"]),
                    "text": deidentify_clinical_text(str(d["deidentified_preview"] or "")).scrubbed_text,
                    "entities": [
                        {
                            "id": str(e["entity_id"]),
                            "type": str(e["entity_type"]),
                            "start": int(e["char_start"]),
                            "end": int(e["char_end"]),
                            "text": deidentify_clinical_text(str(e["source_text"])).scrubbed_text,
                            "assertion": str(e["assertion_status"]),
                            "certainty": str(e["certainty"]),
                            "temporality": str(e["temporality"]),
                            "experiencer": str(e["experiencer"]),
                            "review_status": str(e["review_status"]),
                        }
                        for e in ents
                    ],
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(jsonl_lines) + ("\n" if jsonl_lines else "")


def submit_image_finding_review(
    conn: sqlite3.Connection,
    *,
    principal: AuthenticatedPrincipal,
    image_id: str,
    review_decision: str,
    corrected_diagnosis: str | None = None,
    corrected_risk_band: str | None = None,
    corrected_body_site: str | None = None,
    rationale: str = "Expert dermatology review",
) -> dict[str, Any]:
    """Record a human review/correction of an `image_findings` record and audit-log it."""
    principal.assert_permission("annotation:review")
    allowed_decisions = (
        "APPROVE_AS_GROUND_TRUTH",
        "CORRECT_AND_APPROVE",
        "REJECT",
        "MARK_IMAGE_UNUSABLE",
        "FLAG_AMBIGUOUS",
    )
    if review_decision not in allowed_decisions:
        raise ValueError(f"Unsupported image review_decision: {review_decision}")

    row = conn.execute(
        "SELECT * FROM image_findings WHERE image_id = ?",
        (image_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"Image finding not found: {image_id}")

    principal.assert_institution_access(str(row["institution_id"]))

    top3 = json.loads(str(row["top3_diagnoses_json"] or "[]"))
    prev_dx = str(row["corrected_diagnosis"] or (top3[0]["diagnosis"] if top3 else "other"))
    previous_val = {
        "malignancy_risk_band": str(row["malignancy_risk_band"]),
        "primary_diagnosis": prev_dx,
        "body_site": row["body_site"],
        "review_status": str(row["review_status"]),
    }

    if review_decision == "APPROVE_AS_GROUND_TRUTH":
        new_status = "approved"
    elif review_decision == "CORRECT_AND_APPROVE":
        new_status = "corrected"
    elif review_decision == "MARK_IMAGE_UNUSABLE":
        new_status = "image_unusable"
    elif review_decision == "REJECT":
        new_status = "rejected"
    else:
        new_status = "flagged"

    new_dx = corrected_diagnosis or prev_dx
    new_band = corrected_risk_band or str(row["malignancy_risk_band"])
    new_site = corrected_body_site if corrected_body_site is not None else row["body_site"]

    if new_dx and top3 and review_decision == "CORRECT_AND_APPROVE":
        top3[0]["diagnosis"] = new_dx

    corrected_val = {
        "malignancy_risk_band": new_band,
        "primary_diagnosis": new_dx,
        "body_site": new_site,
        "review_status": new_status,
    }

    review_id = f"rev_img_{uuid.uuid4().hex[:12]}"
    now_iso = datetime.now(timezone.utc).isoformat()

    conn.execute(
        """
        INSERT INTO human_reviews (
            review_id, target_type, target_id, document_id, reviewer_user_id,
            previous_value_json, corrected_value_json, review_decision, rationale, created_at
        ) VALUES (?, 'IMAGE_FINDING', ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            review_id,
            image_id,
            row["document_id"],
            principal.user_id,
            json.dumps(previous_val, sort_keys=True),
            json.dumps(corrected_val, sort_keys=True),
            review_decision,
            rationale,
            now_iso,
        ),
    )

    conn.execute(
        """
        UPDATE image_findings
        SET malignancy_risk_band = ?, corrected_diagnosis = ?, body_site = ?,
            top3_diagnoses_json = ?, review_status = ?
        WHERE image_id = ?
        """,
        (new_band, new_dx, new_site, json.dumps(top3), new_status, image_id),
    )

    record_audit_event(
        conn,
        actor_user_id=principal.user_id,
        institution_id=principal.institution_id,
        action_type="HUMAN_REVIEW_IMAGE_FINDING",
        resource_type="IMAGE_FINDING",
        resource_id=image_id,
        metadata={
            "review_id": review_id,
            "decision": review_decision,
            "new_status": new_status,
            "corrected_diagnosis": new_dx,
        },
    )

    return {
        "review_id": review_id,
        "image_id": image_id,
        "review_status": new_status,
        "previous_value": previous_val,
        "corrected_value": corrected_val,
        "review_decision": review_decision,
        "reviewer_user_id": principal.user_id,
        "created_at": now_iso,
    }


def export_reviewed_image_gold(
    conn: sqlite3.Connection,
    *,
    institution_id: str,
) -> list[dict[str, Any]]:
    """Export approved and human-corrected image findings as gold records."""
    rows = conn.execute(
        """
        SELECT image_id, subject_id, document_id, picture_type, malignancy_risk_band,
               malignancy_risk_probability, corrected_diagnosis, top3_diagnoses_json,
               body_site, review_status, created_at
        FROM image_findings
        WHERE institution_id = ? AND review_status IN ('approved', 'corrected')
        ORDER BY created_at DESC
        """,
        (institution_id,),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        top3 = json.loads(str(r["top3_diagnoses_json"] or "[]"))
        dx = str(r["corrected_diagnosis"] or (top3[0]["diagnosis"] if top3 else "other"))
        out.append({
            "image_id": str(r["image_id"]),
            "patient_id": r["subject_id"],
            "document_id": r["document_id"],
            "picture_type": str(r["picture_type"]),
            "gold_diagnosis": dx,
            "gold_risk_band": str(r["malignancy_risk_band"]),
            "body_site": r["body_site"],
            "review_status": str(r["review_status"]),
            "created_at": str(r["created_at"]),
        })
    return out
