"""Approach 2: Classical ML (TF-IDF + LinearSVC token tagger) and CRF baseline (sklearn-crfsuite)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn_crfsuite
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from medresearch_platform.approaches.context_negex import apply_context_and_relations
from medresearch_platform.approaches.ocr_augment import build_ocr_augmented_train_set
from medresearch_platform.approaches.self_supervised import (
    SelfSupervisedMedicalEmbeddings,
    load_optional_self_supervised,
)
from medresearch_platform.approaches.tokenizer import (
    Token,
    align_entities_to_bio,
    decode_bio_to_entities,
    tokenize_with_offsets,
)
from medresearch_platform.data.schema import AnnotatedDocument
from medresearch_platform.eval_harness.metrics import DocumentPrediction
from medresearch_platform.hybrid.persistence import (
    compute_corpus_hash,
    git_commit_hash,
    load_crf_model,
    save_crf_model,
    verify_manifest,
    write_manifest,
)


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


_MED_DISORDER_SUFFIXES = (
    "itis", "ite", "osis", "ose", "oma", "ome", "pathy", "pathie",
    "emia", "émie", "algia", "algie", "penia", "pénie", "megaly", "mégalie",
)
_MED_PROC_SUFFIXES = (
    "ectomy", "ectomie", "scopy", "scopie", "graphy", "graphie",
    "plasty", "plastie", "otomy", "otomie", "gram", "gramme", "metry", "métrie",
)
_MED_DRUG_SUFFIXES = (
    "cillin", "cilline", "mycin", "mycine", "statin", "statine", "pril",
    "olol", "prazole", "sartan", "dipine", "mab", "nib", "vir", "pam", "lam",
)
_MED_UNIT_TOKENS = frozenset(
    {
        "mg", "g", "kg", "mcg", "ug", "µg", "ml", "l", "dl", "iu", "ui",
        "mmol", "meq", "mmhg", "bpm", "%", "cp", "comprimé", "comprimés",
        "gélule", "gélules", "sachet", "sachets", "goutte", "gouttes",
        "milligramme", "milligrammes", "gramme", "grammes",
    }
)


def _token_window_features(
    tokens: list[Token],
    idx: int,
    *,
    sem_model: SelfSupervisedMedicalEmbeddings | None = None,
) -> dict[str, Any]:
    tok = tokens[idx].text
    low = tok.lower()
    feats: dict[str, Any] = {
        "bias": 1.0,
        "word.lower": low,
        "word.shape": _word_shape(tok),
        "word.isupper": tok.isupper(),
        "word.istitle": tok.istitle(),
        "word.isdigit": tok.isdigit(),
        "word.has_digit": any(c.isdigit() for c in tok),
        "word.has_hyphen": "-" in tok,
        "word.has_slash": "/" in tok,
        "word.len_bin": min(len(tok), 15),
        "word.pref2": low[:2],
        "word.pref3": low[:3],
        "word.pref4": low[:4],
        "word.suff2": low[-2:],
        "word.suff3": low[-3:],
        "word.suff4": low[-4:],
        "word.suff5": low[-5:],
        "word.is_med_disorder_suff": low.endswith(_MED_DISORDER_SUFFIXES),
        "word.is_med_proc_suff": low.endswith(_MED_PROC_SUFFIXES),
        "word.is_med_drug_suff": low.endswith(_MED_DRUG_SUFFIXES),
        "word.is_unit": low in _MED_UNIT_TOKENS,
    }
    if sem_model is not None:
        cl = sem_model.get_clusters(low)
        if cl is not None:
            feats["w0.sem64"] = f"c{cl[0]}"
            feats["w0.sem16"] = f"g{cl[1]}"

    for offset in (-2, -1, 1, 2):
        j = idx + offset
        key = f"w{offset:+d}"
        if 0 <= j < len(tokens):
            ntok = tokens[j].text
            nlow = ntok.lower()
            feats[f"{key}.lower"] = nlow
            feats[f"{key}.shape"] = _word_shape(ntok)
            if offset in (-1, 1):
                feats[f"{key}.is_unit"] = nlow in _MED_UNIT_TOKENS
                feats[f"{key}.has_digit"] = any(c.isdigit() for c in ntok)
                if sem_model is not None:
                    ncl = sem_model.get_clusters(nlow)
                    if ncl is not None:
                        feats[f"{key}.sem64"] = f"c{ncl[0]}"
        else:
            feats[f"{key}.bos_eos"] = True

    if idx > 0:
        feats["bg.prev"] = f"{tokens[idx - 1].text.lower()}|{low}"
    if idx + 1 < len(tokens):
        feats["bg.next"] = f"{low}|{tokens[idx + 1].text.lower()}"
    return feats


def _build_doc_vectorizer() -> FeatureUnion:
    return FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=4000, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=4000, sublinear_tf=True)),
        ]
    )


def _to_int32_csr(mat: Any) -> Any:
    mat.indices = mat.indices.astype(np.int32, copy=False)
    mat.indptr = mat.indptr.astype(np.int32, copy=False)
    return mat


def _doc_classifier_fit_predict(
    train_docs: list[AnnotatedDocument],
    eval_docs: list[AnnotatedDocument],
    seed: int,
) -> list[str]:
    vec = _build_doc_vectorizer()
    x_tr = _to_int32_csr(vec.fit_transform([d.text for d in train_docs]))
    y_tr = [d.doc_type for d in train_docs]
    clf = LinearSVC(random_state=seed, class_weight="balanced")
    clf.fit(x_tr, y_tr)
    x_ev = _to_int32_csr(vec.transform([d.text for d in eval_docs]))
    return [str(p) for p in clf.predict(x_ev)]


def _select_train_docs(
    train_docs: list[AnnotatedDocument], seed: int, cfg: dict[str, Any]
) -> list[AnnotatedDocument]:
    raw_max = cfg.get("max_train_docs", None)
    if raw_max is None:
        return train_docs
    rng = np.random.default_rng(seed)
    shuffled = list(train_docs)
    rng.shuffle(shuffled)
    return shuffled[: int(raw_max)]


def predict_classical_ml(
    train_docs: list[AnnotatedDocument],
    eval_docs: list[AnnotatedDocument],
    seed: int,
    cfg: dict[str, Any],
) -> list[DocumentPrediction]:
    """Approach 2a: Token window features + LinearSVC BIO tagger + TF-IDF doc classifier."""
    tr_subset = _select_train_docs(train_docs, seed, cfg)
    x_feats: list[dict[str, Any]] = []
    y_tags: list[str] = []
    for d in tr_subset:
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

    pred_doc_types = _doc_classifier_fit_predict(tr_subset, eval_docs, seed)
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


class ClassicalCRFModel:
    """Saveable/loadable Linear-chain CRF + TF-IDF doc-type classifier."""

    def __init__(self, *, seed: int = 42, cfg: dict[str, Any] | None = None) -> None:
        self.seed = seed
        self.cfg = cfg or {}
        self.crf = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=float(self.cfg.get("c1", 0.1)),
            c2=float(self.cfg.get("c2", 0.1)),
            max_iterations=int(self.cfg.get("max_iterations", 80)),
            all_possible_transitions=True,
        )
        self.doc_vec: Any = _build_doc_vectorizer()
        self.doc_clf = LinearSVC(random_state=seed, class_weight="balanced")
        self._single_dtype: str | None = None
        self.sem_model: SelfSupervisedMedicalEmbeddings | None = (
            load_optional_self_supervised(self.cfg.get("self_supervised_path"))
            if bool(self.cfg.get("use_self_supervised", False))
            else None
        )
        self.corpus_id = str(self.cfg.get("dataset_dir", "in_memory"))
        self.corpus_hash = ""
        self.manifest: dict[str, Any] = {}
        self._fitted = False

    def fit(self, train_docs: list[AnnotatedDocument]) -> ClassicalCRFModel:
        tr_subset = _select_train_docs(train_docs, self.seed, self.cfg)
        self.corpus_hash = compute_corpus_hash(tr_subset)
        ocr_ratio = float(self.cfg.get("ocr_augment_ratio", 0.0))
        fit_docs = (
            build_ocr_augmented_train_set(tr_subset, ratio=ocr_ratio, seed=self.seed)
            if ocr_ratio > 0.0
            else tr_subset
        )
        x_train = []
        y_train = []
        for d in fit_docs:
            toks = tokenize_with_offsets(d.text)
            tags = align_entities_to_bio(toks, d.entities)
            x_train.append([_token_window_features(toks, i, sem_model=self.sem_model) for i in range(len(toks))])
            y_train.append(tags)
        self.crf.fit(x_train, y_train)

        x_doc = _to_int32_csr(self.doc_vec.fit_transform([d.text for d in tr_subset]))
        dtypes = [d.doc_type for d in tr_subset]
        if len(set(dtypes)) >= 2:
            self.doc_clf.fit(x_doc, dtypes)
            self._single_dtype = None
        else:
            self._single_dtype = dtypes[0] if dtypes else "discharge_summary"
        self._fitted = True
        return self

    def predict_document(self, d: AnnotatedDocument) -> DocumentPrediction:
        if not self._fitted:
            raise RuntimeError("ClassicalCRFModel must be fitted before predict_document()")
        if self._single_dtype is not None:
            dt_pred = self._single_dtype
        else:
            x_ev = _to_int32_csr(self.doc_vec.transform([d.text]))
            dt_pred = str(self.doc_clf.predict(x_ev)[0])
        toks = tokenize_with_offsets(d.text)
        if not toks:
            return DocumentPrediction(d.doc_id, d.language, d.doc_type, predicted_doc_type=dt_pred)
        feats = [_token_window_features(toks, i, sem_model=self.sem_model) for i in range(len(toks))]
        pred_tags = list(self.crf.predict_single(feats))
        marginals = self.crf.predict_marginals_single(feats)
        confs = [float(m.get(t, 0.75)) for m, t in zip(marginals, pred_tags, strict=True)]
        ents = decode_bio_to_entities(d.text, toks, pred_tags, confs)
        enriched, relations, timex = apply_context_and_relations(d, ents)
        return DocumentPrediction(
            doc_id=d.doc_id,
            language=d.language,
            doc_type=d.doc_type,
            predicted_doc_type=dt_pred,
            entities=enriched,
            relations=relations,
            timex=timex,
        )

    def predict_docs(self, eval_docs: list[AnnotatedDocument]) -> list[DocumentPrediction]:
        return [self.predict_document(d) for d in eval_docs]

    def save(self, model_dir: Path | str, *, corpus_id: str | None = None) -> Path:
        if not self._fitted:
            raise RuntimeError("ClassicalCRFModel must be fitted before save()")
        out_dir = Path(model_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_crf_model(self.crf, out_dir / "crf.crfsuite")
        joblib.dump(
            {
                "doc_vec": self.doc_vec,
                "doc_clf": self.doc_clf,
                "single_dtype": self._single_dtype,
            },
            out_dir / "doc_clf.joblib",
        )
        manifest_path = write_manifest(
            out_dir,
            {
                "model_type": "classical_crf",
                "seed": self.seed,
                "cfg": self.cfg,
                "corpus_id": corpus_id or self.corpus_id,
                "corpus_hash": self.corpus_hash,
                "git_commit": git_commit_hash(),
            },
        )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return out_dir

    @classmethod
    def load(cls, model_dir: Path | str) -> ClassicalCRFModel:
        in_dir = Path(model_dir)
        manifest = verify_manifest(in_dir)
        if manifest.get("model_type") != "classical_crf":
            raise ValueError(f"Expected model_type='classical_crf', got {manifest.get('model_type')!r}")
        model = cls(seed=int(manifest["seed"]), cfg=dict(manifest.get("cfg", {})))
        model.corpus_id = str(manifest.get("corpus_id", "unknown"))
        model.corpus_hash = str(manifest.get("corpus_hash", ""))
        model.manifest = manifest
        model.crf = load_crf_model(in_dir / "crf.crfsuite")
        doc_bundle = joblib.load(in_dir / "doc_clf.joblib")
        model.doc_vec = doc_bundle["doc_vec"]
        model.doc_clf = doc_bundle["doc_clf"]
        model._single_dtype = doc_bundle["single_dtype"]
        model._fitted = True
        return model


def predict_classical_crf(
    train_docs: list[AnnotatedDocument],
    eval_docs: list[AnnotatedDocument],
    seed: int,
    cfg: dict[str, Any],
) -> list[DocumentPrediction]:
    """Approach 2b: Linear-chain CRF baseline (sklearn-crfsuite) with window features."""
    return ClassicalCRFModel(seed=seed, cfg=cfg).fit(train_docs).predict_docs(eval_docs)
