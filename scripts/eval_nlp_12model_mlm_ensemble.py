"""Benchmark the 12-Model Hybrid MLM + Multi-Capacity Hospital NLP Ensemble
(3x 96d + 3x 128d + 3x 160d + 3x From-Scratch MLM-Pretrained 160d/192d + CRF v2)
on open_clinical_test, realistic_llm_80, and synth_unseen_terms.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from medresearch_platform.approaches.classical_ml import ClassicalCRFModel
from medresearch_platform.approaches.context_negex import apply_context_and_relations
from medresearch_platform.approaches.torch_ner import (
    PyTorchCharCNNBiLSTMCRF,
    ensemble_torch_ner_predictions,
)
from medresearch_platform.eval_harness.metrics import DocumentPrediction, Evaluator
from medresearch_platform.eval_harness.runner import _load_jsonl
from scripts.train_hospital_models import _eval_split, _merge_hospital_ensemble


def main() -> None:
    t0 = time.perf_counter()
    models_root = Path("models")
    models: list[PyTorchCharCNNBiLSTMCRF] = []
    for sd in range(1, 13):
        d = models_root / f"hospital_char_cnn_bilstm_crf_seed{sd}"
        if not d.exists():
            raise FileNotFoundError(f"Missing model directory: {d}")
        m = PyTorchCharCNNBiLSTMCRF.load(d, device="cpu")
        models.append(m)
    print(f"Loaded {len(models)} SHA-256 verified hospital NLP models (seeds 1..12).", flush=True)

    synth_train = _load_jsonl(Path("data/processed/v3_1/train.jsonl"))
    synth_unseen = _load_jsonl(Path("data/processed/v3_1/test_unseen_terms.jsonl"))
    open_train = _load_jsonl(Path("data/processed/open_clinical_plus_biomed/train.jsonl"))
    open_test_all = _load_jsonl(Path("data/processed/open_clinical_plus_biomed/test.jsonl"))
    open_test = [d for d in open_test_all if d.source in {"maccrobat2020", "pxcorpus", "e3c_layer1"}]
    llm_80 = _load_jsonl(Path("data/processed/realistic_llm/eval.jsonl"))

    ev = Evaluator()
    crf_synth = ClassicalCRFModel(seed=1, cfg={"use_self_supervised": True, "ocr_augment_ratio": 0.30}).fit(synth_train)
    open_clin_train = [d for d in open_train if d.source in {"maccrobat2020", "pxcorpus", "e3c_layer1"}]
    crf_open = ClassicalCRFModel(seed=1, cfg={"use_self_supervised": True, "ocr_augment_ratio": 0.15}).fit(open_clin_train)

    splits = {
        "open_clinical_test": open_test,
        "realistic_llm_80": llm_80,
        "synth_unseen_terms": synth_unseen,
    }

    report: dict[str, Any] = {
        "evaluated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": "post-hoc (test seen) — 12-Model Hybrid MLM + Multi-Capacity Hospital Ensemble (3x96d + 3x128d + 3x160d + 3x MLM-pretrained 160d/192d, 8/12 supermajority + CRF v2)",
        "pretrained_weights_used": False,
        "self_supervised_mlm_from_scratch": True,
        "ensemble_architecture": {
            "total_models": len(models),
            "models_96d": 3,
            "models_128d_kaggle_cuda": 3,
            "models_160d_kaggle_cuda": 3,
            "models_mlm_160d_192d_kaggle_cuda": 3,
            "voting_rule": "8/12 supermajority (or mean_conf >= 0.94) + CRF v2 merge",
        },
        "splits": {},
    }

    for sp_name, sp_docs in splits.items():
        ts = time.perf_counter()
        p_t: list[DocumentPrediction] = []
        p_e: list[DocumentPrediction] = []
        all_stds: list[float] = []
        high_unc_cnt = 0
        total_ent_cnt = 0

        for d in sp_docs:
            cp = (crf_open if sp_name == "open_clinical_test" else crf_synth).predict_document(d)
            te = ensemble_torch_ner_predictions(models, d.text)
            for e in te:
                all_stds.append(float(e.confidence_variance))
                total_ent_cnt += 1
                if e.high_uncertainty:
                    high_unc_cnt += 1
            ten, tr, tx = apply_context_and_relations(d, te)
            p_t.append(DocumentPrediction(d.doc_id, d.language, d.doc_type, cp.predicted_doc_type, ten, tr, tx))
            ee = _merge_hospital_ensemble(d, te, cp.entities)
            een, er, ex = apply_context_and_relations(d, ee)
            p_e.append(DocumentPrediction(d.doc_id, d.language, d.doc_type, cp.predicted_doc_type, een, er, ex))

        mt = _eval_split(ev, sp_docs, p_t)
        me = _eval_split(ev, sp_docs, p_e)
        me["uncertainty_stats"] = {
            "total_torch_entities": total_ent_cnt,
            "high_uncertainty_count": high_unc_cnt,
            "high_uncertainty_rate": round(high_unc_cnt / max(1, total_ent_cnt), 4),
            "mean_confidence_variance": round(float(np.mean(all_stds)), 4) if all_stds else 0.0,
        }
        report["splits"][sp_name] = {
            "torch_12model_only": mt,
            "hospital_ensemble_12model": me,
        }
        print(
            f"[{sp_name:20s}] ({time.perf_counter() - ts:.1f}s) "
            f"torch={mt['strict_f1']:.4f}/{mt['lenient_f1']:.4f} | "
            f"ens={me['strict_f1']:.4f}/{me['lenient_f1']:.4f} "
            f"(P={me['precision']:.4f} R={me['recall']:.4f}, high_unc={me['uncertainty_stats']['high_uncertainty_rate']*100:.1f}%)",
            flush=True,
        )

    report["elapsed_seconds"] = round(time.perf_counter() - t0, 2)
    out_path = Path("docs/results/hospital_12model_mlm_ensemble_benchmark.json")
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {out_path} in {report['elapsed_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
