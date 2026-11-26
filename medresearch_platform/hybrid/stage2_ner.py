"""Stage 2: Entity Extraction — BiLSTM-CRF merged with high-precision rule/lexicon layer."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import sklearn_crfsuite

from medresearch_platform.approaches.classical_ml import _token_window_features
from medresearch_platform.approaches.neural_seq import (
    NeuralBiLSTMCRFTagger,
    _AdamW,
    _compute_class_weights,
    _encode_docs,
    _ortho_features,
    _softmax,
)
from medresearch_platform.approaches.ocr_augment import build_ocr_augmented_train_set
from medresearch_platform.approaches.rule_based import predict_rule_based
from medresearch_platform.approaches.self_supervised import (
    SelfSupervisedMedicalEmbeddings,
    load_optional_self_supervised,
)
from medresearch_platform.approaches.tokenizer import (
    TrainOnlyVocabulary,
    align_entities_to_bio,
    decode_bio_to_entities,
    tokenize_with_offsets,
)
from medresearch_platform.data.schema import AnnotatedDocument
from medresearch_platform.eval_harness.metrics import PredictedEntity
from medresearch_platform.security.person_age_detector import (
    detect_age_spans,
    detect_person_spans,
    span_overlaps,
)


@dataclass(frozen=True)
class CandidateEntity:
    entity: PredictedEntity
    source_stage: str  # "bilstm_crf", "rule_layer", or "bilstm_crf+rule_layer"


_NON_ENTITY_AGE_NOUNS: frozenset[str] = frozenset(
    {
        "boy",
        "girl",
        "infant",
        "toddler",
        "child",
        "teenager",
        "newborn",
        "neonate",
        "nourrisson",
        "enfant",
        "garçon",
        "fille",
        "bébé",
        "adolescent",
        "adolescente",
    }
)

_TIME_UNIT_WORDS: frozenset[str] = frozenset(
    {
        "semaine",
        "semaines",
        "week",
        "weeks",
        "jour",
        "jours",
        "day",
        "days",
        "mois",
        "month",
        "months",
        "an",
        "ans",
        "year",
        "years",
        "année",
        "années",
        "heure",
        "heures",
        "hour",
        "hours",
        "matin",
        "soir",
        "nuit",
    }
)

_ASSERTION_CUE_WORDS: frozenset[str] = frozenset(
    {
        "ruled",
        "excluded",
        "écartée",
        "écarté",
        "éliminée",
        "éliminé",
        "exclu",
        "exclue",
        "denied",
        "denies",
        "nie",
        "absence",
        "sans",
        "without",
    }
)

_DURATION_AFTER_PREP_RE = re.compile(
    r"\b(?:depuis|pendant|durant|for|since|during(?:\s+the\s+last)?)\s+"
    r"((?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fourteen|fifteen|twenty|thirty|"
    r"un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|quelques|several|a\s+few)\s+"
    r"(?:heures?|jours?|semaines?|mois|ans?|années?|hours?|days?|weeks?|months?|years?))\b",
    re.IGNORECASE | re.UNICODE,
)

_NON_MEDICATION_VERBS: frozenset[str] = frozenset(
    {
        "took", "take", "takes", "taking", "started", "stopped", "prescribed", "given", "received",
        "prend", "prends", "pris", "prise", "reçoit", "reçu", "débuté", "arrêté",
    }
)

_TRAILING_AUX_VERBS: frozenset[str] = frozenset(
    {"does", "do", "did", "is", "was", "are", "were", "has", "have", "had"}
)

_CLOSED_CLASS_NON_ENTITIES: frozenset[str] = frozenset(
    {
        "his", "her", "their", "our", "my", "your", "its", "the", "this", "that", "these", "those",
        "son", "sa", "ses", "leur", "leurs", "mon", "ma", "mes", "le", "la", "les", "un", "une", "des",
        "normal", "normale", "normaux", "normales",
    }
)


def _is_disallowed_entity(ent: PredictedEntity) -> bool:
    low = ent.text.strip().lower()
    if not low:
        return True
    if low in _NON_ENTITY_AGE_NOUNS:
        return True
    if low in _ASSERTION_CUE_WORDS:
        return True
    if ent.type in {"DIAGNOSIS", "SYMPTOM", "MEDICATION", "DATE", "ANATOMY", "PROCEDURE"} and low in _CLOSED_CLASS_NON_ENTITIES:
        return True
    if ent.type == "MEDICATION" and low in _NON_MEDICATION_VERBS:
        return True
    if ent.type in {"MEDICATION", "SYMPTOM", "DIAGNOSIS"} and low in _TIME_UNIT_WORDS:
        return True
    if ent.type == "ANATOMY":
        toks = low.split()
        if any(t in _TIME_UNIT_WORDS for t in toks):
            return True
        if toks and toks[-1] in _TRAILING_AUX_VERBS:
            return True
    return False


class Stage2EntityExtractor:
    """Trains BiLSTM-CRF on train_docs and merges spans with the rule/lexicon layer."""

    def __init__(
        self,
        *,
        hidden_dim: int = 32,
        epochs: int = 10,
        lr: float = 0.018,
        max_train_docs: int | None = None,
        use_crf_ensemble: bool = False,
        use_self_supervised: bool = False,
        self_supervised_path: str | None = None,
        ocr_augment_ratio: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.max_train_docs = max_train_docs
        self.use_crf_ensemble = use_crf_ensemble
        self.use_self_supervised = use_self_supervised
        self.self_supervised_path = self_supervised_path
        self.ocr_augment_ratio = ocr_augment_ratio
        self.seed = seed
        self.vocab = TrainOnlyVocabulary(max_size=6000, min_freq=1)
        self.model: NeuralBiLSTMCRFTagger | None = None
        self.crf_model: sklearn_crfsuite.CRF | None = None
        self.sem_model: SelfSupervisedMedicalEmbeddings | None = (
            load_optional_self_supervised(self_supervised_path) if use_self_supervised else None
        )
        self.heldout_es_doc_ids: set[str] = set()

    def fit(self, train_docs: list[AnnotatedDocument]) -> Stage2EntityExtractor:
        self.vocab.fit(train_docs)
        rng = np.random.default_rng(self.seed)
        shuffled = [train_docs[int(i)] for i in rng.permutation(len(train_docs))]
        subset = shuffled if self.max_train_docs is None else shuffled[: self.max_train_docs]
        n_es = max(1, int(0.12 * len(subset)))
        es_docs = subset[:n_es]
        self.heldout_es_doc_ids = {d.doc_id for d in es_docs}
        tr_docs = subset[n_es:] if len(subset) > n_es else subset
        if self.ocr_augment_ratio > 0.0:
            tr_docs = build_ocr_augmented_train_set(tr_docs, ratio=self.ocr_augment_ratio, seed=self.seed)

        tr_seqs = _encode_docs(tr_docs, self.vocab)
        es_seqs = _encode_docs(es_docs, self.vocab)
        cw = _compute_class_weights(tr_seqs)

        model = NeuralBiLSTMCRFTagger(len(self.vocab), emb_dim=24, hidden_dim=self.hidden_dim, seed=self.seed)
        if self.sem_model is not None:
            proj, matched = self.sem_model.project_to_vocab(self.vocab.word_to_id, 24)
            if matched > 0:
                mask = np.linalg.norm(proj, axis=1) > 0.0
                model.params["E"][mask] = proj[mask]
        opt = _AdamW(model.params, lr=self.lr)
        best_loss = float("inf")
        best_params = {k: v.copy() for k, v in model.params.items()}
        stale = 0

        for _ in range(self.epochs):
            for idx in rng.permutation(len(tr_seqs)):
                w_ids, c_feats, y_ids = tr_seqs[int(idx)]
                model.step_sequence(w_ids, c_feats, y_ids, cw, opt)
            es_losses = []
            for w_ids, c_feats, y_ids in es_seqs:
                logits, _ = model.forward_logits(w_ids, c_feats)
                probs = _softmax(logits[: len(y_ids)])
                y_sl = y_ids[: len(probs)]
                es_losses.append(float(-np.mean(cw[y_sl] * np.log(np.maximum(probs[np.arange(len(y_sl)), y_sl], 1e-12)))))
            mean_es = sum(es_losses) / max(1, len(es_losses))
            if mean_es < best_loss - 1e-4:
                best_loss = mean_es
                best_params = {k: v.copy() for k, v in model.params.items()}
                stale = 0
            else:
                stale += 1
                if stale >= 3:
                    break

        model.params = best_params
        self.model = model

        if self.use_crf_ensemble:
            crf_docs = (
                build_ocr_augmented_train_set(subset, ratio=self.ocr_augment_ratio, seed=self.seed + 7)
                if self.ocr_augment_ratio > 0.0
                else subset
            )
            x_crf = []
            y_crf = []
            for d in crf_docs:
                toks = tokenize_with_offsets(d.text)
                if not toks:
                    continue
                x_crf.append([_token_window_features(toks, i, sem_model=self.sem_model) for i in range(len(toks))])
                y_crf.append(align_entities_to_bio(toks, d.entities))
            crf = sklearn_crfsuite.CRF(
                algorithm="lbfgs",
                c1=0.1,
                c2=0.1,
                max_iterations=80,
                all_possible_transitions=True,
            )
            if x_crf:
                crf.fit(x_crf, y_crf)
            self.crf_model = crf

        return self

    def predict(
        self,
        doc: AnnotatedDocument,
        *,
        use_rules_layer: bool = True,
    ) -> list[CandidateEntity]:
        if self.model is None:
            raise RuntimeError("Stage2EntityExtractor must be fitted before predict()")

        # Run person & age detector BEFORE NER so person/age spans can never be clinical entities
        blocked_spans: list[tuple[int, int]] = [
            (sp.start, sp.end) for sp in detect_person_spans(doc.text)
        ] + detect_age_spans(doc.text)

        toks = tokenize_with_offsets(doc.text)
        neural_ents: list[PredictedEntity] = []
        if toks:
            w_ids = np.array([self.vocab.encode_token(t.text) for t in toks], dtype=np.int64)
            c_feats = _ortho_features(toks)
            tags, confs = self.model.predict(w_ids, c_feats)
            neural_ents = decode_bio_to_entities(doc.text, toks, tags, confs)

        merged: dict[tuple[str, int, int], CandidateEntity] = {}
        for e in neural_ents:
            if not span_overlaps(e.start, e.end, blocked_spans):
                merged[(e.type, e.start, e.end)] = CandidateEntity(entity=e, source_stage="bilstm_crf")

        if self.crf_model is not None and toks:
            feats = [_token_window_features(toks, i, sem_model=self.sem_model) for i in range(len(toks))]
            pred_tags = list(self.crf_model.predict_single(feats))
            marginals = self.crf_model.predict_marginals_single(feats)
            confs = [float(m.get(t, 0.75)) for m, t in zip(marginals, pred_tags, strict=True)]
            crf_ents = decode_bio_to_entities(doc.text, toks, pred_tags, confs)
            for ce in crf_ents:
                if span_overlaps(ce.start, ce.end, blocked_spans):
                    continue
                key = (ce.type, ce.start, ce.end)
                if key in merged:
                    prev = merged[key].entity
                    boosted = PredictedEntity(
                        type=prev.type,
                        start=prev.start,
                        end=prev.end,
                        text=prev.text,
                        assertion=prev.assertion,
                        confidence=round(min(0.99, max(prev.confidence, ce.confidence) + 0.05), 4),
                    )
                    merged[key] = CandidateEntity(entity=boosted, source_stage="bilstm_crf+crf")
                else:
                    overlapping = [
                        k
                        for k, m in merged.items()
                        if max(ce.start, m.entity.start) < min(ce.end, m.entity.end)
                    ]
                    if not overlapping:
                        merged[key] = CandidateEntity(entity=ce, source_stage="crf_backoff")
                    elif all(ce.confidence > merged[k].entity.confidence + 0.05 for k in overlapping):
                        for k in overlapping:
                            del merged[k]
                        merged[key] = CandidateEntity(entity=ce, source_stage="crf_override")

        if use_rules_layer:
            rule_pred = predict_rule_based([], [doc], self.seed, {})[0]
            for re_ent in rule_pred.entities:
                if span_overlaps(re_ent.start, re_ent.end, blocked_spans):
                    continue
                key = (re_ent.type, re_ent.start, re_ent.end)
                if key in merged:
                    prev = merged[key].entity
                    boosted = PredictedEntity(
                        type=prev.type,
                        start=prev.start,
                        end=prev.end,
                        text=prev.text,
                        assertion=prev.assertion,
                        confidence=round(min(0.99, max(prev.confidence, re_ent.confidence) + 0.04), 4),
                    )
                    merged[key] = CandidateEntity(entity=boosted, source_stage="bilstm_crf+rule_layer")
                else:
                    overlapping_keys = [
                        k
                        for k, m in merged.items()
                        if max(re_ent.start, m.entity.start) < min(re_ent.end, m.entity.end)
                    ]
                    if not overlapping_keys:
                        merged[key] = CandidateEntity(entity=re_ent, source_stage="rule_layer")
                    elif (
                        re_ent.type in ("MEDICATION", "DOSAGE") or re_ent.confidence >= 0.90
                    ) and all(
                        merged[k].entity.type == "DATE" or merged[k].entity.confidence < 0.65
                        for k in overlapping_keys
                    ):
                        for k in overlapping_keys:
                            del merged[k]
                        merged[key] = CandidateEntity(entity=re_ent, source_stage="rule_layer")

        # Filter disallowed non-clinical tokens (age nouns, bare time words, assertion cue verbs)
        for k in list(merged.keys()):
            if _is_disallowed_entity(merged[k].entity):
                del merged[k]

        # Emit typed DURATION spans after depuis/pendant/for/since when not already covered
        for m in _DURATION_AFTER_PREP_RE.finditer(doc.text):
            ds, de = m.start(1), m.end(1)
            if span_overlaps(ds, de, blocked_spans):
                continue
            overlapping_keys = [
                k for k, c in merged.items() if max(ds, c.entity.start) < min(de, c.entity.end)
            ]
            if any(merged[k].entity.type in {"DURATION", "DATE"} for k in overlapping_keys):
                continue
            for k in overlapping_keys:
                if merged[k].entity.type in {"MEDICATION", "SYMPTOM", "DIAGNOSIS"} or merged[k].entity.text.lower() in _TIME_UNIT_WORDS:
                    del merged[k]
            still_overlapping = any(max(ds, c.entity.start) < min(de, c.entity.end) for c in merged.values())
            if not still_overlapping:
                dur_ent = PredictedEntity(
                    type="DURATION",
                    start=ds,
                    end=de,
                    text=doc.text[ds:de],
                    assertion="present",
                    confidence=0.92,
                )
                merged[("DURATION", ds, de)] = CandidateEntity(entity=dur_ent, source_stage="rule_layer")

        return sorted(merged.values(), key=lambda c: c.entity.start)
