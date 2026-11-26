"""Stage 1: Document Understanding — train-only char n-gram LangID, TF-IDF+LinearSVC doc classifier, sectioning."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from medresearch_platform.data.schema import AnnotatedDocument


def _to_int32_csr(mat: Any) -> Any:
    mat.indices = mat.indices.astype(np.int32, copy=False)
    mat.indptr = mat.indptr.astype(np.int32, copy=False)
    return mat


@dataclass
class DocUnderstandingResult:
    language: str
    language_confidence: float
    doc_type: str
    doc_type_confidence: float
    sections: list[dict[str, Any]]


class Stage1DocumentUnderstanding:
    """Trained strictly on train_docs: char n-gram LangID + TF-IDF LinearSVC doc-type classifier."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.lang_vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=3000)
        self.lang_clf = LogisticRegression(max_iter=300, random_state=seed)
        self.doc_vec: Any = FeatureUnion(
            [
                ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=4000, sublinear_tf=True)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=4000, sublinear_tf=True)),
            ]
        )
        self.doc_clf = LinearSVC(random_state=seed, class_weight="balanced")
        self._fitted = False

    def fit(self, train_docs: list[AnnotatedDocument]) -> Stage1DocumentUnderstanding:
        texts = [d.text for d in train_docs]
        langs = [d.language for d in train_docs]
        dtypes = [d.doc_type for d in train_docs]

        x_lang = _to_int32_csr(self.lang_vec.fit_transform(texts))
        if len(set(langs)) >= 2:
            self.lang_clf.fit(x_lang, langs)
            self._single_lang = None
        else:
            self._single_lang = langs[0] if langs else "en"

        x_doc = _to_int32_csr(self.doc_vec.fit_transform(texts))
        if len(set(dtypes)) >= 2:
            self.doc_clf.fit(x_doc, dtypes)
            self._single_dtype = None
        else:
            self._single_dtype = dtypes[0] if dtypes else "discharge_summary"

        self._fitted = True
        return self

    def predict(self, text: str) -> DocUnderstandingResult:
        if not self._fitted:
            raise RuntimeError("Stage1DocumentUnderstanding must be fitted before predict()")

        if self._single_lang is not None:
            lang, lang_conf = self._single_lang, 1.0
        else:
            x_l = _to_int32_csr(self.lang_vec.transform([text]))
            probs = self.lang_clf.predict_proba(x_l)[0]
            best_i = int(np.argmax(probs))
            lang = str(self.lang_clf.classes_[best_i])
            lang_conf = float(probs[best_i])

        if self._single_dtype is not None:
            dtype, dt_conf = self._single_dtype, 1.0
        else:
            x_d = _to_int32_csr(self.doc_vec.transform([text]))
            dtype = str(self.doc_clf.predict(x_d)[0])
            margins = self.doc_clf.decision_function(x_d)[0]
            dt_conf = float(1.0 / (1.0 + np.exp(-np.max(margins))))

        sections = segment_sections(text)
        return DocUnderstandingResult(
            language=lang,
            language_confidence=round(lang_conf, 4),
            doc_type=dtype,
            doc_type_confidence=round(dt_conf, 4),
            sections=sections,
        )


_SECTION_HDR_RE = re.compile(r"^([A-ZÉÈÊÀÂÎÔÛÇ][A-ZÉÈÊÀÂÎÔÛÇ\s&'\-]{2,45}):?\s*$", re.MULTILINE)


def segment_sections(text: str) -> list[dict[str, Any]]:
    """Identify uppercase clinical section headers and their character spans."""
    matches = list(_SECTION_HDR_RE.finditer(text))
    if not matches:
        return [{"name": "BODY", "start": 0, "end": len(text)}]
    sections: list[dict[str, Any]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append({"name": m.group(1).strip(), "start": start, "end": end})
    return sections
