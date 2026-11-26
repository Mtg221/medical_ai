"""Step 1 (J2): Verify HybridPipeline & ClassicalCRFModel save/load with fresh-process reload."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

from medresearch_platform.approaches.classical_ml import ClassicalCRFModel
from medresearch_platform.data.synthetic_v2 import generate_synthetic_corpus
from medresearch_platform.hybrid.pipeline import HybridPipeline


def _serialize_pred(pred: object) -> dict:
    from medresearch_platform.eval_harness.metrics import DocumentPrediction

    assert isinstance(pred, DocumentPrediction)
    return {
        "doc_id": pred.doc_id,
        "predicted_doc_type": pred.predicted_doc_type,
        "entities": [
            (e.type, e.start, e.end, e.text, e.assertion, round(e.confidence, 6))
            for e in pred.entities
        ],
        "relations": sorted(
            (r.type, r.head_start, r.head_end, r.tail_start, r.tail_end) for r in pred.relations
        ),
        "timex": sorted((t.type, t.start, t.end, t.normalized) for t in pred.timex),
    }


def test_hybrid_v2_save_and_fresh_process_reload_identical_20_docs(tmp_path: Path) -> None:
    corpus = generate_synthetic_corpus(en_count=20, fr_count=20, seed=17)
    train_docs = corpus[:20]
    eval_docs = corpus[20:40]
    assert len(eval_docs) == 20

    cfg = {
        "hidden_dim": 16,
        "epochs": 2,
        "aux_epochs": 2,
        "lr": 0.02,
        "use_crf_ensemble": True,
        "use_rules_layer": True,
        "use_assertion_bilstm": False,
        "use_relation_cnn": True,
        "use_temporal_crf": True,
        "use_transformer_fusion": False,
    }
    pipe = HybridPipeline(seed=1, cfg=cfg).fit(train_docs)
    model_dir = tmp_path / "hybrid_v2_test"
    pipe.save(model_dir, corpus_id="synthetic_small")

    expected = [_serialize_pred(pipe.extract_document(d)[0]) for d in eval_docs]

    eval_path = tmp_path / "eval_docs.json"
    eval_path.write_text(json.dumps([dataclasses.asdict(d) for d in eval_docs]), encoding="utf-8")

    script = (
        "import json; from pathlib import Path; "
        "from medresearch_platform.data.schema import AnnotatedDocument; "
        "from medresearch_platform.hybrid.pipeline import HybridPipeline; "
        f"pipe = HybridPipeline.load(Path({str(model_dir)!r})); "
        f"docs = [AnnotatedDocument(**x) for x in json.loads(Path({str(eval_path)!r}).read_text())]; "
        "out = [{'doc_id': p.doc_id, 'predicted_doc_type': p.predicted_doc_type, "
        "'entities': [[e.type, e.start, e.end, e.text, e.assertion, round(e.confidence, 6)] for e in p.entities], "
        "'relations': sorted([[r.type, r.head_start, r.head_end, r.tail_start, r.tail_end] for r in p.relations]), "
        "'timex': sorted([[t.type, t.start, t.end, t.normalized] for t in p.timex])} "
        "for p in (pipe.extract_document(d)[0] for d in docs)]; "
        "print(json.dumps(out))"
    )
    raw = subprocess.check_output([sys.executable, "-c", script], text=True)
    reloaded = json.loads(raw)
    for item in reloaded:
        item["entities"] = [tuple(e) for e in item["entities"]]
        item["relations"] = [tuple(r) for r in item["relations"]]
        item["timex"] = [tuple(t) for t in item["timex"]]

    assert reloaded == expected


def test_classical_crf_save_and_fresh_process_reload_identical_20_docs(tmp_path: Path) -> None:
    corpus = generate_synthetic_corpus(en_count=20, fr_count=20, seed=19)
    train_docs = corpus[:20]
    eval_docs = corpus[20:40]

    crf_model = ClassicalCRFModel(seed=1, cfg={"max_iterations": 20}).fit(train_docs)
    model_dir = tmp_path / "classical_crf_test"
    crf_model.save(model_dir, corpus_id="synthetic_small")

    expected = [_serialize_pred(p) for p in crf_model.predict_docs(eval_docs)]
    eval_path = tmp_path / "eval_docs.json"
    eval_path.write_text(json.dumps([dataclasses.asdict(d) for d in eval_docs]), encoding="utf-8")

    script = (
        "import json; from pathlib import Path; "
        "from medresearch_platform.data.schema import AnnotatedDocument; "
        "from medresearch_platform.approaches.classical_ml import ClassicalCRFModel; "
        f"m = ClassicalCRFModel.load(Path({str(model_dir)!r})); "
        f"docs = [AnnotatedDocument(**x) for x in json.loads(Path({str(eval_path)!r}).read_text())]; "
        "out = [{'doc_id': p.doc_id, 'predicted_doc_type': p.predicted_doc_type, "
        "'entities': [[e.type, e.start, e.end, e.text, e.assertion, round(e.confidence, 6)] for e in p.entities], "
        "'relations': sorted([[r.type, r.head_start, r.head_end, r.tail_start, r.tail_end] for r in p.relations]), "
        "'timex': sorted([[t.type, t.start, t.end, t.normalized] for t in p.timex])} "
        "for p in m.predict_docs(docs)]; "
        "print(json.dumps(out))"
    )
    raw = subprocess.check_output([sys.executable, "-c", script], text=True)
    reloaded = json.loads(raw)
    for item in reloaded:
        item["entities"] = [tuple(e) for e in item["entities"]]
        item["relations"] = [tuple(r) for r in item["relations"]]
        item["timex"] = [tuple(t) for t in item["timex"]]

    assert reloaded == expected

    # Tamper check
    (model_dir / "crf.crfsuite").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        ClassicalCRFModel.load(model_dir)
