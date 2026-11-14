"""Approach 2: TF-IDF + Classical ML BIO Sequence Tagger (scikit-learn)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from medresearch_platform.ai.terminology import BilingualTerminologyRegistry
from medresearch_platform.extraction.contract import ExtractionResult, records_from_bio_tags
from medresearch_platform.extraction.text import TAGS, Token, normalize_token, spans_to_bio, tokenize


def _token_shape(word: str) -> str:
    out: list[str] = []
    for ch in word[:12]:
        if ch.isdigit():
            out.append("d")
        elif ch.isalpha():
            out.append("X" if ch.isupper() else "x")
        else:
            out.append(ch)
    return "".join(out)


def _token_context_features(tokens: list[Token], idx: int) -> dict[str, Any]:
    curr = normalize_token(tokens[idx].text)
    prev1 = normalize_token(tokens[idx - 1].text) if idx > 0 else "<BOS>"
    prev2 = normalize_token(tokens[idx - 2].text) if idx > 1 else "<BOS2>"
    next1 = normalize_token(tokens[idx + 1].text) if idx + 1 < len(tokens) else "<EOS>"
    next2 = normalize_token(tokens[idx + 2].text) if idx + 2 < len(tokens) else "<EOS2>"
    raw = tokens[idx].text
    return {
        "curr": curr,
        "prev1": prev1,
        "prev2": prev2,
        "next1": next1,
        "next2": next2,
        "prev1_curr": f"{prev1}|{curr}",
        "curr_next1": f"{curr}|{next1}",
        "is_digit": raw.replace(".", "", 1).isdigit(),
        "has_digit": any(c.isdigit() for c in raw),
        "is_title": raw[:1].isupper(),
        "shape": _token_shape(raw),
        "prefix3": curr[:3],
        "suffix3": curr[-3:],
    }


@dataclass
class ClassicalMLExtractor:
    """TF-IDF char-ngram + lexical window LogisticRegression BIO tagger."""

    random_state: int = 42
    max_iter: int = 300
    C: float = 2.0
    model_name: str = "ClassicalTFIDFExtractor"
    model_version: str = "1.0.0"

    def __post_init__(self) -> None:
        self.dict_vec = DictVectorizer(sparse=True)
        self.tfidf_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            lowercase=True,
            sublinear_tf=True,
            max_features=2500,
        )
        self.classifier = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.terminology = BilingualTerminologyRegistry()
        self.is_fitted = False
        self.parameter_count = 0

    def _featurize_tokens(self, tokens: list[Token], *, fit: bool) -> sparse.csr_matrix:
        dicts = [_token_context_features(tokens, i) for i in range(len(tokens))]
        surfaces = [normalize_token(t.text) for t in tokens]
        if fit:
            x_dict = self.dict_vec.fit_transform(dicts)
            x_tfidf = self.tfidf_vec.fit_transform(surfaces)
        else:
            x_dict = self.dict_vec.transform(dicts)
            x_tfidf = self.tfidf_vec.transform(surfaces)
        return sparse.hstack([x_dict, x_tfidf], format="csr")

    def fit(self, corpus: list[tuple[str, list[tuple[int, int, str]]]]) -> dict[str, Any]:
        """Train from scratch on [(text, [(char_start, char_end, entity_type), ...]), ...]."""
        all_dicts: list[dict[str, Any]] = []
        all_surfaces: list[str] = []
        all_tags: list[str] = []

        for text, spans in corpus:
            toks = tokenize(text)
            if not toks:
                continue
            tags = spans_to_bio(toks, spans)
            for i, tok in enumerate(toks):
                all_dicts.append(_token_context_features(toks, i))
                all_surfaces.append(normalize_token(tok.text))
                all_tags.append(tags[i])

        x_dict = self.dict_vec.fit_transform(all_dicts)
        x_tfidf = self.tfidf_vec.fit_transform(all_surfaces)
        x_mat = sparse.hstack([x_dict, x_tfidf], format="csr")
        y_arr = np.asarray(all_tags, dtype=object)

        self.classifier.fit(x_mat, y_arr)
        self.is_fitted = True
        self.parameter_count = int(self.classifier.coef_.size + self.classifier.intercept_.size)
        return {
            "num_tokens": len(all_tags),
            "num_features": int(x_mat.shape[1]),
            "num_classes": int(len(self.classifier.classes_)),
            "parameter_count": self.parameter_count,
        }

    def predict_bio(self, text: str) -> tuple[list[Token], list[str], list[float]]:
        if not self.is_fitted:
            raise RuntimeError("ClassicalMLExtractor must be fitted before prediction.")
        toks = tokenize(text)
        if not toks:
            return [], [], []
        x_mat = self._featurize_tokens(toks, fit=False)
        probs = self.classifier.predict_proba(x_mat)
        best_idx = np.argmax(probs, axis=1)
        classes = self.classifier.classes_
        pred_tags = [str(classes[i]) for i in best_idx]
        confs = [float(probs[row, best_idx[row]]) for row in range(len(toks))]
        return toks, pred_tags, confs

    def extract(
        self,
        text: str,
        *,
        source_record_id: str = "record-0",
        document_id: str = "doc-0",
        language: str = "en",
        page_number: int = 1,
    ) -> ExtractionResult:
        toks, tags, confs = self.predict_bio(text)
        return records_from_bio_tags(
            text=text,
            tokens=toks,
            tags=tags,
            token_confidences=confs,
            source_record_id=source_record_id,
            document_id=document_id,
            language=language,
            model_name=self.model_name,
            model_version=self.model_version,
            provenance={
                "approach": "classical_tfidf_logreg",
                "feature_spaces": ["lexical_window_dict", "char_wb_tfidf_2_4"],
                "parameter_count": self.parameter_count,
                "pretrained_weights_used": False,
                "possible_tags": len(TAGS),
            },
            terminology=self.terminology,
            page_number=page_number,
        )
