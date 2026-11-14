"""Approach 2: Classical ML (TF-IDF + LinearSVC token tagger) and CRF baseline (sklearn-crfsuite)."""

from __future__ import annotations

from typing import Any

import numpy as np
import sklearn_crfsuite
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from medresearch_platform.approaches.context_negex import apply_context_and_relations
from medresearch_platform.approaches.tokenizer import (
    Token,
    align_entities_to_bio,
    decode_bio_to_entities,
    tokenize_with_offsets,
)
from medresearch_platform.data.schema import AnnotatedDocument
from medresearch_platform.eval_harness.metrics import DocumentPrediction


def _word_shape(word: str) -> str:
    chars: list[str] = []
    for ch in word[:12]:
        if ch.isupper():
            chars.append("X")
        elif ch.islower():
            chars.append("x")
        elif ch.isdigit():
            chars.append("d")
        else:
            chars.append(ch)
    return "".join(chars)


def _token_window_features(tokens: list[Token], idx: int) -> dict[str, Any]:
    tok = tokens[idx].text
    feats: dict[str, Any] = {
        "bias": 1.0,
        "word.lower": tok.lower(),
        "word.shape": _word_shape(tok),
        "word.isupper": tok.isupper(),
        "word.istitle": tok.istitle(),
        "word.isdigit": tok.isdigit(),
        "word.has_digit": any(c.isdigit() for c in tok),
        "word.pref2": tok[:2].lower(),
        "word.pref3": tok[:3].lower(),
        "word.suff2": tok[-2:].lower(),
        "word.suff3": tok[-3:].lower(),
    }
    for offset in (-2, -1, 1, 2):
        j = idx + offset
        key = f"w{offset:+d}"
        if 0 <= j < len(tokens):
            ntok = tokens[j].text
            feats[f"{key}.lower"] = ntok.lower()
            feats[f"{key}.shape"] = _word_shape(ntok)
        else:
            feats[f"{key}.bos_eos"] = True
    return feats


def _to_int32_csr(mat: Any) -> Any:
    mat.indices = mat.indices.astype(np.int32, copy=False)
    mat.indptr = mat.indptr.astype(np.int32, copy=False)
    return mat


def _doc_classifier_fit_predict(
    train_docs: list[AnnotatedDocument],
    eval_docs: list[AnnotatedDocument],
    seed: int,
) -> list[str]:
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=4000, sublinear_tf=True)
    x_tr = _to_int32_csr(vec.fit_transform([d.text for d in train_docs]))
    y_tr = [d.doc_type for d in train_docs]
    clf = LinearSVC(random_state=seed, class_weight="balanced")
    clf.fit(x_tr, y_tr)
    x_ev = _to_int32_csr(vec.transform([d.text for d in eval_docs]))
    return [str(p) for p in clf.predict(x_ev)]


def predict_classical_ml(
    train_docs: list[AnnotatedDocument],
    eval_docs: list[AnnotatedDocument],
    seed: int,
    cfg: dict[str, Any],
) -> list[DocumentPrediction]:
    """Approach 2a: Token window features + LinearSVC BIO tagger + TF-IDF doc classifier."""
    x_feats: list[dict[str, Any]] = []
    y_tags: list[str] = []
    for d in train_docs:
        toks = tokenize_with_offsets(d.text)
        tags = align_entities_to_bio(toks, d.entities)
        for i in range(len(toks)):
            x_feats.append(_token_window_features(toks, i))
            y_tags.append(tags[i])

    vec = DictVectorizer(sparse=True)
    x_mat = _to_int32_csr(vec.fit_transform(x_feats))
    c_val = float(cfg.get("c", 1.0))
    clf = LinearSVC(C=c_val, random_state=seed)
    clf.fit(x_mat, y_tags)

    pred_doc_types = _doc_classifier_fit_predict(train_docs, eval_docs, seed)
    preds: list[DocumentPrediction] = []
    for d, dt_pred in zip(eval_docs, pred_doc_types, strict=True):
        toks = tokenize_with_offsets(d.text)
        if not toks:
            preds.append(DocumentPrediction(d.doc_id, d.language, d.doc_type, predicted_doc_type=dt_pred))
            continue
        ev_feats = [_token_window_features(toks, i) for i in range(len(toks))]
        ev_mat = _to_int32_csr(vec.transform(ev_feats))
        pred_tags = [str(t) for t in clf.predict(ev_mat)]
        margins = clf.decision_function(ev_mat)
        confs = [
            float(1.0 / (1.0 + np.exp(-np.max(row)))) if margins.ndim == 2 else 0.75
            for row in (margins if margins.ndim == 2 else margins[:, None])
        ]
        ents = decode_bio_to_entities(d.text, toks, pred_tags, confs)
        enriched, relations, timex = apply_context_and_relations(d, ents)
        preds.append(
            DocumentPrediction(
                doc_id=d.doc_id,
                language=d.language,
                doc_type=d.doc_type,
                predicted_doc_type=dt_pred,
                entities=enriched,
                relations=relations,
                timex=timex,
            )
        )
    return preds


def predict_classical_crf(
    train_docs: list[AnnotatedDocument],
    eval_docs: list[AnnotatedDocument],
    seed: int,
    cfg: dict[str, Any],
) -> list[DocumentPrediction]:
    """Approach 2b: Linear-chain CRF baseline (sklearn-crfsuite) with window features."""
    x_train = []
    y_train = []
    for d in train_docs:
        toks = tokenize_with_offsets(d.text)
        tags = align_entities_to_bio(toks, d.entities)
        x_train.append([_token_window_features(toks, i) for i in range(len(toks))])
        y_train.append(tags)

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=float(cfg.get("c1", 0.1)),
        c2=float(cfg.get("c2", 0.1)),
        max_iterations=int(cfg.get("max_iterations", 80)),
        all_possible_transitions=True,
    )
    crf.fit(x_train, y_train)

    pred_doc_types = _doc_classifier_fit_predict(train_docs, eval_docs, seed)
    preds: list[DocumentPrediction] = []
    for d, dt_pred in zip(eval_docs, pred_doc_types, strict=True):
        toks = tokenize_with_offsets(d.text)
        if not toks:
            preds.append(DocumentPrediction(d.doc_id, d.language, d.doc_type, predicted_doc_type=dt_pred))
            continue
        feats = [_token_window_features(toks, i) for i in range(len(toks))]
        pred_tags = list(crf.predict_single(feats))
        marginals = crf.predict_marginals_single(feats)
        confs = [float(m.get(t, 0.75)) for m, t in zip(marginals, pred_tags, strict=True)]
        ents = decode_bio_to_entities(d.text, toks, pred_tags, confs)
        enriched, relations, timex = apply_context_and_relations(d, ents)
        preds.append(
            DocumentPrediction(
                doc_id=d.doc_id,
                language=d.language,
                doc_type=d.doc_type,
                predicted_doc_type=dt_pred,
                entities=enriched,
                relations=relations,
                timex=timex,
            )
        )
    return preds
