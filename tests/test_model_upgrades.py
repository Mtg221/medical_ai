"""Unit tests for from-scratch self-supervised embeddings, OCR augmentation, and Stage 3b gating."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from medresearch_platform.approaches.classical_ml import ClassicalCRFModel
from medresearch_platform.approaches.ocr_augment import (
    augment_document_ocr_noise,
    build_ocr_augmented_train_set,
)
from medresearch_platform.approaches.self_supervised import SelfSupervisedMedicalEmbeddings
from medresearch_platform.approaches.tokenizer import TrainOnlyVocabulary
from medresearch_platform.data.synthetic_v3 import generate_v3_corpus_splits
from medresearch_platform.hybrid.stage1_doc_understanding import Stage1DocumentUnderstanding
from medresearch_platform.hybrid.stage3_context_relations import BiLSTMAssertionClassifier


def test_self_supervised_embeddings_fit_save_load_and_project(tmp_path: Path) -> None:
    texts = [
        "Patient presents with acute pneumonia treated with amoxicillin 500 mg daily.",
        "Chest radiograph showed right lower lobe pneumonia and mild fever.",
        "Le patient présente une pneumonie aiguë traitée par amoxicilline 1 g par jour.",
        "Absence de fièvre ; diabète de type 2 équilibré sous metformine 500 mg.",
        "Type 2 diabetes managed with metformin 500 mg twice daily and HbA1c monitoring.",
        "Amoxicillin prescribed for community acquired pneumonia with fever.",
    ] * 4
    model = SelfSupervisedMedicalEmbeddings.fit_from_texts(
        texts, max_vocab=200, min_count=2, dim=8, n_clusters_fine=8, n_clusters_coarse=4, seed=7
    )
    assert model.metadata["pretrained_weights_used"] is False
    assert model.dim == 8
    cl = model.get_clusters("pneumonia")
    assert cl is not None and 0 <= cl[0] < 8 and 0 <= cl[1] < 4

    out = model.save(tmp_path / "ss_test.npz")
    loaded = SelfSupervisedMedicalEmbeddings.load(out)
    assert loaded.get_clusters("pneumonia") == cl
    proj, matched = loaded.project_to_vocab({"<PAD>": 0, "<UNK>": 1, "pneumonia": 2, "metformin": 3}, 12)
    assert matched == 2
    assert proj.shape == (4, 12)
    assert float(np.linalg.norm(proj[2])) > 0.0


def test_ocr_noise_augmentation_preserves_exact_spans() -> None:
    splits = generate_v3_corpus_splits(
        split_counts_per_lang={"train": 4, "val": 2, "test_iid": 2, "test_unseen_terms": 2, "test_unseen_templates": 2},
        seed=5,
    )
    rng = np.random.default_rng(5)
    for doc in splits["train"]:
        aug = augment_document_ocr_noise(doc, rng, char_error_rate=0.10, accent_drop_prob=0.8)
        assert len(aug.text) == len(doc.text)
        assert aug.validate_spans() == []

    combined = build_ocr_augmented_train_set(splits["train"], ratio=0.5, seed=5)
    assert len(combined) == len(splits["train"]) + len(splits["train"]) // 2


def test_upgraded_crf_stage1_and_calibrated_assertion_end_to_end() -> None:
    splits = generate_v3_corpus_splits(
        split_counts_per_lang={"train": 10, "val": 4, "test_iid": 2, "test_unseen_terms": 2, "test_unseen_templates": 2},
        seed=13,
    )
    s1 = Stage1DocumentUnderstanding(seed=13).fit(splits["train"])
    pred_s1 = s1.predict(splits["val"][0].text)
    assert pred_s1.language in ("en", "fr")

    crf = ClassicalCRFModel(seed=13, cfg={"ocr_augment_ratio": 0.25}).fit(splits["train"])
    pred_crf = crf.predict_document(splits["val"][0])
    assert pred_crf.doc_id == splits["val"][0].doc_id

    vocab = TrainOnlyVocabulary(max_size=1000, min_freq=1).fit(splits["train"])
    clf = BiLSTMAssertionClassifier(len(vocab), seed=13).fit(splits["train"], vocab, epochs=2, seed=13)
    lbl = clf.predict_calibrated(
        splits["val"][0].text,
        splits["val"][0].entities[0].start,
        splits["val"][0].entities[0].end,
        vocab,
        fallback_label="present",
    )
    assert isinstance(lbl, str) and len(lbl) > 0
