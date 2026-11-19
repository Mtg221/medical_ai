"""Tests for from-scratch PyTorch Char-CNN + BiLSTM + Partial-Label Marginal CRF."""

from __future__ import annotations

from pathlib import Path

import pytest

from medresearch_platform.approaches.tokenizer import TAG_TO_ID
from medresearch_platform.approaches.torch_ner import (
    ALL_ENTITY_TYPES,
    NUM_TAGS,
    PyTorchCharCNNBiLSTMCRF,
    _build_torch_ner_module,
    build_allowed_tag_mask,
    infer_annotated_types,
    is_torch_available,
)
from medresearch_platform.data.schema import AnnotatedDocument, EntitySpan


def test_partial_label_allowed_mask_allows_unannotated_types_on_o_tokens() -> None:
    y_ids = [TAG_TO_ID["O"], TAG_TO_ID["B-MEDICATION"], TAG_TO_ID["O"]]
    # Full schema: 'O' allows only 'O'
    full_mask = build_allowed_tag_mask(y_ids, ALL_ENTITY_TYPES)
    assert int(full_mask[0].sum()) == 1
    assert full_mask[0, TAG_TO_ID["O"]]
    assert int(full_mask[1].sum()) == 1
    assert full_mask[1, TAG_TO_ID["B-MEDICATION"]]

    # Partial schema (only MEDICATION annotated): 'O' allows O + B/I of DIAGNOSIS, SYMPTOM, etc.
    partial_mask = build_allowed_tag_mask(y_ids, frozenset({"MEDICATION"}))
    assert partial_mask[0, TAG_TO_ID["O"]]
    assert partial_mask[0, TAG_TO_ID["B-DIAGNOSIS"]]
    assert not partial_mask[0, TAG_TO_ID["B-MEDICATION"]]
    # Labeled token still locked strictly to B-MEDICATION
    assert int(partial_mask[1].sum()) == 1
    assert partial_mask[1, TAG_TO_ID["B-MEDICATION"]]


def test_infer_annotated_types_maps_external_corpora() -> None:
    px_doc = AnnotatedDocument("p1", "pat1", "fr", "prescription", "2025-01-01", "pxcorpus", "flagyl 500 mg")
    assert infer_annotated_types(px_doc) == frozenset({"MEDICATION", "DOSAGE", "DURATION", "FREQUENCY", "ROUTE"})
    synth_doc = AnnotatedDocument("s1", "pat2", "en", "consultation", "2025-01-01", "synth_v3", "fever")
    assert infer_annotated_types(synth_doc) == ALL_ENTITY_TYPES


@pytest.mark.skipif(not is_torch_available(), reason="PyTorch not installed in active interpreter")
def test_torch_partial_crf_loss_nonnegative_and_save_load(tmp_path: Path) -> None:
    import torch

    net = _build_torch_ner_module(
        vocab_size=32,
        char_vocab_size=32,
        word_emb_dim=8,
        char_emb_dim=4,
        char_filters=4,
        cluster_vocab_size=8,
        cluster_emb_dim=4,
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
    )
    emissions = torch.randn(2, 5, NUM_TAGS)
    seq_mask = torch.tensor([[True, True, True, True, True], [True, True, True, False, False]])
    y = [TAG_TO_ID["O"], TAG_TO_ID["B-MEDICATION"], TAG_TO_ID["O"], TAG_TO_ID["O"], TAG_TO_ID["O"]]
    full_m = torch.from_numpy(build_allowed_tag_mask(y, ALL_ENTITY_TYPES)).unsqueeze(0).repeat(2, 1, 1)
    part_m = torch.from_numpy(build_allowed_tag_mask(y, frozenset({"MEDICATION"}))).unsqueeze(0).repeat(2, 1, 1)

    loss_full = float(net.partial_crf_loss(emissions, full_m, seq_mask).item())
    loss_part = float(net.partial_crf_loss(emissions, part_m, seq_mask).item())
    assert loss_full >= 0.0
    assert loss_part >= 0.0
    # Marginalizing unannotated types increases constrained partition -> strictly <= full loss
    assert loss_part <= loss_full + 1e-5

    docs = [
        AnnotatedDocument(
            doc_id=f"d{i}",
            patient_id=f"p{i}",
            language="en",
            doc_type="prescription",
            document_date="2025-03-01",
            source="synth_v3" if i % 2 == 0 else "pxcorpus",
            text="Started metformin 500 mg daily for diabetes.",
            entities=[
                EntitySpan(id="e1", type="MEDICATION", start=8, end=17, text="metformin"),
                EntitySpan(id="e2", type="DOSAGE", start=18, end=24, text="500 mg"),
            ],
        )
        for i in range(12)
    ]
    model = PyTorchCharCNNBiLSTMCRF(
        word_emb_dim=16,
        char_emb_dim=8,
        char_filters=8,
        hidden_dim=24,
        num_layers=1,
        epochs=2,
        batch_size=4,
        use_self_supervised=False,
        seed=7,
    ).fit(docs)
    saved = model.save(tmp_path / "torch_ner")
    loaded = PyTorchCharCNNBiLSTMCRF.load(saved)
    assert loaded.trained_weights_sha256 == model.trained_weights_sha256
    preds_a = model.predict_entities(docs[0].text)
    preds_b = loaded.predict_entities(docs[0].text)
    assert [(e.type, e.start, e.end) for e in preds_a] == [(e.type, e.start, e.end) for e in preds_b]
