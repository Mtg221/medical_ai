"""Stage 2: Entity Extraction — BiLSTM-CRF merged with high-precision rule/lexicon layer."""

from __future__ import annotations

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
from medresearch_platform.approaches.rule_based import predict_rule_based
from medresearch_platform.approaches.tokenizer import (
    TrainOnlyVocabulary,
    align_entities_to_bio,
    decode_bio_to_entities,
    tokenize_with_offsets,
)
from medresearch_platform.data.schema import AnnotatedDocument
from medresearch_platform.eval_harness.metrics import PredictedEntity


@dataclass(frozen=True)
class CandidateEntity:
    entity: PredictedEntity
    source_stage: str  # "bilstm_crf", "rule_layer", or "bilstm_crf+rule_layer"


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
        seed: int = 42,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.max_train_docs = max_train_docs
        self.use_crf_ensemble = use_crf_ensemble
        self.seed = seed
        self.vocab = TrainOnlyVocabulary(max_size=6000, min_freq=1)
        self.model: NeuralBiLSTMCRFTagger | None = None
        self.crf_model: sklearn_crfsuite.CRF | None = None

    def fit(self, train_docs: list[AnnotatedDocument]) -> Stage2EntityExtractor:
        self.vocab.fit(train_docs)
        rng = np.random.default_rng(self.seed)
        shuffled = list(train_docs)
        rng.shuffle(shuffled)
        subset = shuffled if self.max_train_docs is None else shuffled[: self.max_train_docs]
        n_es = max(1, int(0.12 * len(subset)))
        es_docs = subset[:n_es]
        tr_docs = subset[n_es:] if len(subset) > n_es else subset

        tr_seqs = _encode_docs(tr_docs, self.vocab)
        es_seqs = _encode_docs(es_docs, self.vocab)
        cw = _compute_class_weights(tr_seqs)

        model = NeuralBiLSTMCRFTagger(len(self.vocab), emb_dim=24, hidden_dim=self.hidden_dim, seed=self.seed)
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
            x_crf = []
            y_crf = []
            for d in subset:
                toks = tokenize_with_offsets(d.text)
                if not toks:
                    continue
                x_crf.append([_token_window_features(toks, i) for i in range(len(toks))])
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

        toks = tokenize_with_offsets(doc.text)
        neural_ents: list[PredictedEntity] = []
        if toks:
            w_ids = np.array([self.vocab.encode_token(t.text) for t in toks], dtype=np.int64)
            c_feats = _ortho_features(toks)
            tags, confs = self.model.predict(w_ids, c_feats)
            neural_ents = decode_bio_to_entities(doc.text, toks, tags, confs)

        merged: dict[tuple[str, int, int], CandidateEntity] = {}
        for e in neural_ents:
            merged[(e.type, e.start, e.end)] = CandidateEntity(entity=e, source_stage="bilstm_crf")

        if self.crf_model is not None and toks:
            feats = [_token_window_features(toks, i) for i in range(len(toks))]
            pred_tags = list(self.crf_model.predict_single(feats))
            marginals = self.crf_model.predict_marginals_single(feats)
            confs = [float(m.get(t, 0.75)) for m, t in zip(marginals, pred_tags, strict=True)]
            crf_ents = decode_bio_to_entities(doc.text, toks, pred_tags, confs)
            for ce in crf_ents:
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

        if not use_rules_layer:
            return sorted(merged.values(), key=lambda c: c.entity.start)

        rule_pred = predict_rule_based([], [doc], self.seed, {})[0]

        for re_ent in rule_pred.entities:
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
                # Add rule match only if it does not overlap an existing neural span
                overlaps = any(
                    max(re_ent.start, m.entity.start) < min(re_ent.end, m.entity.end)
                    for m in merged.values()
                )
                if not overlaps:
                    merged[key] = CandidateEntity(entity=re_ent, source_stage="rule_layer")

        return sorted(merged.values(), key=lambda c: c.entity.start)
