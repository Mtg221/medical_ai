"""Human correction, inter-annotator agreement (Cohen's kappa), and adjudication (Section 16)."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from medresearch_platform.security.audit import record_audit_event
from medresearch_platform.security.auth import AuthenticatedPrincipal


def submit_entity_human_review(
    conn: sqlite3.Connection,
    *,
    principal: AuthenticatedPrincipal,
    entity_id: str,
    review_decision: str,
    corrected_entity_type: str | None = None,
    corrected_assertion_status: str | None = None,
    corrected_temporal_status: str | None = None,
    corrected_normalized_text: str | None = None,
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

    previous_val = {
        "entity_type": str(row["entity_type"]),
        "normalized_text": str(row["normalized_text"]),
        "assertion_status": str(row["assertion_status"]),
        "temporal_status": str(row["temporal_status"]),
        "review_status": str(row["review_status"]),
    }

    new_ent_type = corrected_entity_type or previous_val["entity_type"]
    new_norm = corrected_normalized_text or previous_val["normalized_text"]
    new_ass = corrected_assertion_status or previous_val["assertion_status"]
    new_tmp = corrected_temporal_status or previous_val["temporal_status"]

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
        "assertion_status": new_ass,
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
        SET entity_type = ?, normalized_text = ?, assertion_status = ?, temporal_status = ?, review_status = ?
        WHERE entity_id = ?
        """,
        (new_ent_type, new_norm, new_ass, new_tmp, new_status, entity_id),
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
        "previous": previous_val,
        "updated": corrected_val,
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
