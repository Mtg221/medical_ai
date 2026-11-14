"""Approaches 5 & 6: Longitudinal Timeline, Contradiction Detection, Knowledge Graph & Hybrid Retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import math
import sqlite3
from typing import Any
import uuid
import numpy as np

from medresearch_platform.ai.random_init import (
    InitMethod,
    RandomInitManifest,
    build_random_init_manifest,
    initialize_tensor,
)


@dataclass(frozen=True)
class TimelineEvent:
    event_id: str
    subject_id: str
    document_id: str
    event_date_iso: str
    event_category: str
    summary_text: str
    assertion_status: str
    temporal_status: str
    derivation_type: str
    confidence: float
    uncertainty: float
    contradiction_flag: bool
    contradiction_note: str | None


def build_and_persist_subject_timeline(
    conn: sqlite3.Connection,
    subject_id: str,
) -> list[TimelineEvent]:
    """Reconstruct a chronological, evidence-backed timeline for a subject and flag contradictions."""
    conn.execute("DELETE FROM longitudinal_events WHERE subject_id = ?", (subject_id,))
    rows = conn.execute(
        """
        SELECT
            e.entity_id,
            e.document_id,
            e.entity_type,
            e.source_text,
            e.normalized_text,
            e.assertion_status,
            e.temporal_status,
            e.medication_status,
            e.confidence,
            e.uncertainty,
            d.encounter_id,
            COALESCE(enc.encounter_date_iso, substr(d.created_at, 1, 10)) AS event_date_iso
        FROM extracted_entities e
        JOIN documents d ON d.document_id = e.document_id
        LEFT JOIN encounters enc ON enc.encounter_id = d.encounter_id
        WHERE d.subject_id = ?
          AND e.entity_type IN ('DIAGNOSIS', 'SYMPTOM', 'MEDICATION', 'LAB_TEST', 'PROCEDURE')
        ORDER BY event_date_iso ASC, e.char_start ASC
        """,
        (subject_id,),
    ).fetchall()

    # Track prior assertion states per normalized concept to detect cross-document contradictions
    concept_history: dict[str, list[tuple[str, str, str]]] = {}
    for row in rows:
        norm = str(row["normalized_text"]).lower()
        concept_history.setdefault(norm, []).append(
            (str(row["document_id"]), str(row["assertion_status"]), str(row["medication_status"]))
        )

    events: list[TimelineEvent] = []
    for row in rows:
        norm = str(row["normalized_text"]).lower()
        history = concept_history.get(norm, [])
        assertions_seen = {h[1] for h in history}
        med_states_seen = {h[2] for h in history if h[2] != "NOT_APPLICABLE"}

        contradiction_flag = False
        contradiction_note: str | None = None
        if "AFFIRMED" in assertions_seen and "NEGATED" in assertions_seen:
            contradiction_flag = True
            contradiction_note = (
                f"Cross-document assertion divergence on '{norm}': both AFFIRMED and NEGATED mentions recorded."
            )
        elif "TAKEN" in med_states_seen and "DISCONTINUED" in med_states_seen:
            contradiction_flag = True
            contradiction_note = (
                f"Medication evolution/discrepancy on '{norm}': transitions between TAKEN and DISCONTINUED."
            )

        event_id = f"evt_{uuid.uuid4().hex[:14]}"
        summary = f"{row['entity_type']}: {row['normalized_text']} ({row['assertion_status']})"
        if str(row["medication_status"]) != "NOT_APPLICABLE":
            summary += f" [med_state={row['medication_status']}]"

        evt = TimelineEvent(
            event_id=event_id,
            subject_id=subject_id,
            document_id=str(row["document_id"]),
            event_date_iso=str(row["event_date_iso"]),
            event_category=str(row["entity_type"]),
            summary_text=summary,
            assertion_status=str(row["assertion_status"]),
            temporal_status=str(row["temporal_status"]),
            derivation_type="EXTRACTED",
            confidence=float(row["confidence"]),
            uncertainty=float(row["uncertainty"]),
            contradiction_flag=contradiction_flag,
            contradiction_note=contradiction_note,
        )
        events.append(evt)

        conn.execute(
            """
            INSERT INTO longitudinal_events (
                event_id, subject_id, encounter_id, document_id, entity_id,
                event_category, summary_text, assertion_status, temporal_status,
                event_date_iso, derivation_type, confidence, uncertainty,
                contradiction_flag, contradiction_note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evt.event_id,
                evt.subject_id,
                row["encounter_id"],
                evt.document_id,
                row["entity_id"],
                evt.event_category,
                evt.summary_text,
                evt.assertion_status,
                evt.temporal_status,
                evt.event_date_iso,
                evt.derivation_type,
                evt.confidence,
                evt.uncertainty,
                1 if evt.contradiction_flag else 0,
                evt.contradiction_note,
            ),
        )

    return events


class CustomNeuralRetrievalScorer:
    """Approach 6: Hybrid Lexical + Terminology + Custom Random-Initialized Neural Retrieval Engine."""

    architecture_name = "CustomHybridRetrievalNet"
    architecture_version = "1.0.0"

    def __init__(self, seed: int = 20261112, init_method: InitMethod = "xavier_uniform") -> None:
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.params = {
            "proj_w": initialize_tensor(rng, (128, 32), init_method),
            "proj_b": initialize_tensor(rng, (32,), "zeros"),
        }
        self.init_manifest: RandomInitManifest = build_random_init_manifest(
            architecture_name=self.architecture_name,
            architecture_version=self.architecture_version,
            init_method=init_method,
            seed=seed,
            parameters=self.params,
        )

    def embed_text(self, text: str) -> np.ndarray:
        """Compute a 32-dim L2-normalized embedding from character n-gram buckets through random-init projection."""
        buckets = np.zeros((1, 128), dtype=np.float32)
        clean = " ".join(text.lower().split())
        for i in range(max(1, len(clean) - 2)):
            tri = clean[i : i + 3]
            idx = sum(ord(c) * (29 ** k) for k, c in enumerate(tri)) % 128
            buckets[0, idx] += 1.0
        emb = np.tanh(np.matmul(buckets, self.params["proj_w"]) + self.params["proj_b"])[0]
        norm = float(np.linalg.norm(emb))
        return emb / norm if norm > 1e-8 else emb

    def search_documents(
        self,
        conn: sqlite3.Connection,
        *,
        institution_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Execute permission-scoped hybrid lexical + structured entity + neural similarity retrieval."""
        q_tokens = {t.strip(".,:;()").lower() for t in query.split() if len(t.strip(".,:;()")) >= 2}
        q_vec = self.embed_text(query)

        docs = conn.execute(
            """
            SELECT d.document_id, d.original_filename, d.detected_language, d.specialty,
                   d.document_type, d.deidentified_preview
            FROM documents d
            WHERE d.institution_id = ?
            """,
            (institution_id,),
        ).fetchall()

        results: list[dict[str, Any]] = []
        for doc in docs:
            doc_id = str(doc["document_id"])
            pages = conn.execute(
                "SELECT raw_text FROM document_pages WHERE document_id = ? ORDER BY page_number ASC",
                (doc_id,),
            ).fetchall()
            full_text = "\n".join(str(p["raw_text"]) for p in pages)
            entities = conn.execute(
                "SELECT normalized_text, assertion_status, entity_type FROM extracted_entities WHERE document_id = ?",
                (doc_id,),
            ).fetchall()

            doc_tokens = {t.strip(".,:;()").lower() for t in full_text.split() if len(t.strip(".,:;()")) >= 2}
            entity_terms = {str(e["normalized_text"]).lower() for e in entities}

            lexical_overlap = len(q_tokens & doc_tokens) / max(1, len(q_tokens))
            structured_hit = 1.0 if any(qt in term for qt in q_tokens for term in entity_terms) else 0.0
            d_vec = self.embed_text(full_text)
            neural_sim = float(max(0.0, np.dot(q_vec, d_vec)))

            hybrid_score = round(0.55 * lexical_overlap + 0.30 * structured_hit + 0.15 * neural_sim, 4)
            if hybrid_score > 0.05:
                results.append({
                    "document_id": doc_id,
                    "original_filename": str(doc["original_filename"]),
                    "detected_language": str(doc["detected_language"]),
                    "specialty": str(doc["specialty"]),
                    "hybrid_score": hybrid_score,
                    "ranking_explanation": {
                        "lexical_overlap": round(lexical_overlap, 4),
                        "structured_terminology_match": round(structured_hit, 4),
                        "custom_neural_similarity": round(neural_sim, 4),
                        "medical_equivalence_disclaimer": "Search similarity reflects lexical/feature overlap, not clinical equivalence.",
                    },
                    "preview": str(doc["deidentified_preview"])[:220],
                })

        results.sort(key=lambda r: r["hybrid_score"], reverse=True)
        return results[:top_k]


def evaluate_retrieval_metrics(
    ranked_doc_ids: list[str],
    relevant_doc_ids: set[str],
    k: int = 5,
) -> dict[str, float]:
    """Compute Precision@k, Recall@k, MRR, and nDCG@k (Sections 13 & 20)."""
    top_k = ranked_doc_ids[:k]
    hits_at_k = sum(1 for d in top_k if d in relevant_doc_ids)
    precision_at_k = hits_at_k / max(1, k)
    recall_at_k = hits_at_k / max(1, len(relevant_doc_ids))

    rr = 0.0
    for rank_idx, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            rr = 1.0 / rank_idx
            break

    dcg = 0.0
    for idx, doc_id in enumerate(top_k, start=1):
        if doc_id in relevant_doc_ids:
            dcg += 1.0 / math.log2(idx + 1)

    ideal_hits = min(k, len(relevant_doc_ids))
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    return {
        f"precision_at_{k}": round(precision_at_k, 4),
        f"recall_at_{k}": round(recall_at_k, 4),
        "mrr": round(rr, 4),
        f"ndcg_at_{k}": round(ndcg, 4),
    }
