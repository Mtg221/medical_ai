"""Inter-Annotator Agreement (IAA) metrics for double-annotated bilingual medical corpora."""

from __future__ import annotations

from collections import Counter
from typing import Any

from medresearch_platform.data.schema import ASSERTION_CLASSES, AnnotatedDocument


def cohens_kappa(labels_a: list[str], labels_b: list[str], categories: tuple[str, ...] | None = None) -> float:
    """Compute unweighted Cohen's kappa between two aligned categorical label sequences."""
    if len(labels_a) != len(labels_b) or not labels_a:
        return 0.0
    n = len(labels_a)
    cats = list(categories) if categories is not None else sorted(set(labels_a) | set(labels_b))
    p_o = sum(1 for a, b in zip(labels_a, labels_b, strict=True) if a == b) / n
    ca = Counter(labels_a)
    cb = Counter(labels_b)
    p_e = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    if abs(1.0 - p_e) < 1e-12:
        return 1.0 if abs(1.0 - p_o) < 1e-12 else 0.0
    return round((p_o - p_e) / (1.0 - p_e), 4)


def _pair_f1(matches: int, count_a: int, count_b: int) -> float:
    denom = count_a + count_b
    if denom == 0:
        return 1.0
    return round(2.0 * matches / denom, 4)


def compute_document_pair_iaa(
    pairs: list[tuple[AnnotatedDocument, AnnotatedDocument]],
) -> dict[str, Any]:
    """Compute strict/lenient entity F1, assertion Cohen's kappa, relation F1, and TIMEX F1."""
    if not pairs:
        return {
            "double_annotated_docs": 0,
            "entity_strict_f1": 0.0,
            "entity_lenient_f1": 0.0,
            "assertion_cohens_kappa": 0.0,
            "assertion_agreement": 0.0,
            "assertion_matched_support": 0,
            "relation_strict_f1": 0.0,
            "timex_strict_f1": 0.0,
            "per_language": {},
        }

    def _agg(sub_pairs: list[tuple[AnnotatedDocument, AnnotatedDocument]]) -> dict[str, Any]:
        ent_a_total = 0
        ent_b_total = 0
        ent_strict_match = 0
        ent_lenient_match = 0
        assert_a: list[str] = []
        assert_b: list[str] = []
        rel_a_total = 0
        rel_b_total = 0
        rel_match = 0
        tx_a_total = 0
        tx_b_total = 0
        tx_match = 0

        for doc_a, doc_b in sub_pairs:
            ent_a_total += len(doc_a.entities)
            ent_b_total += len(doc_b.entities)

            map_a = {(e.type, e.start, e.end): e for e in doc_a.entities}
            map_b = {(e.type, e.start, e.end): e for e in doc_b.entities}
            common = set(map_a) & set(map_b)
            ent_strict_match += len(common)
            for k in sorted(common):
                assert_a.append(map_a[k].assertion)
                assert_b.append(map_b[k].assertion)

            used_b: set[int] = set()
            for ea in doc_a.entities:
                for j, eb in enumerate(doc_b.entities):
                    if j in used_b or ea.type != eb.type:
                        continue
                    if max(ea.start, eb.start) < min(ea.end, eb.end):
                        used_b.add(j)
                        ent_lenient_match += 1
                        break

            id_span_a = {e.id: (e.start, e.end) for e in doc_a.entities}
            id_span_b = {e.id: (e.start, e.end) for e in doc_b.entities}
            rels_a = {
                (r.type, id_span_a[r.head], id_span_a[r.tail])
                for r in doc_a.relations
                if r.head in id_span_a and r.tail in id_span_a
            }
            rels_b = {
                (r.type, id_span_b[r.head], id_span_b[r.tail])
                for r in doc_b.relations
                if r.head in id_span_b and r.tail in id_span_b
            }
            rel_a_total += len(rels_a)
            rel_b_total += len(rels_b)
            rel_match += len(rels_a & rels_b)

            txs_a = {(t.type, t.start, t.end) for t in doc_a.timex}
            txs_b = {(t.type, t.start, t.end) for t in doc_b.timex}
            tx_a_total += len(txs_a)
            tx_b_total += len(txs_b)
            tx_match += len(txs_a & txs_b)

        assert_acc = (
            round(sum(1 for a, b in zip(assert_a, assert_b, strict=True) if a == b) / len(assert_a), 4)
            if assert_a
            else 0.0
        )
        return {
            "double_annotated_docs": len(sub_pairs),
            "entity_strict_f1": _pair_f1(ent_strict_match, ent_a_total, ent_b_total),
            "entity_lenient_f1": _pair_f1(ent_lenient_match, ent_a_total, ent_b_total),
            "assertion_cohens_kappa": cohens_kappa(assert_a, assert_b, ASSERTION_CLASSES),
            "assertion_agreement": assert_acc,
            "assertion_matched_support": len(assert_a),
            "relation_strict_f1": _pair_f1(rel_match, rel_a_total, rel_b_total),
            "relation_counts": {"annotator_a": rel_a_total, "annotator_b": rel_b_total, "matched": rel_match},
            "timex_strict_f1": _pair_f1(tx_match, tx_a_total, tx_b_total),
            "timex_counts": {"annotator_a": tx_a_total, "annotator_b": tx_b_total, "matched": tx_match},
        }

    overall = _agg(pairs)
    langs = sorted({a.language for a, _ in pairs})
    overall["per_language"] = {
        lang: _agg([(a, b) for a, b in pairs if a.language == lang]) for lang in langs
    }
    return overall
