"""Benchmark all NLP model upgrades (morphology, E3C self-supervised, OCR aug, word+char Stage 1, calibrated Stage 3b)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from medresearch_platform.approaches.classical_ml import ClassicalCRFModel
from medresearch_platform.approaches.neural_seq import predict_bilstm_crf, predict_transformer
from medresearch_platform.approaches.ocr_augment import augment_document_ocr_noise
from medresearch_platform.eval_harness.metrics import Evaluator
from medresearch_platform.eval_harness.runner import _load_jsonl
from medresearch_platform.hybrid.pipeline import HybridPipeline


def main() -> None:
    t0 = time.perf_counter()
    ev = Evaluator()
    open_train = _load_jsonl(Path("data/processed/open_clinical/train.jsonl"))
    open_test = _load_jsonl(Path("data/processed/open_clinical/test.jsonl"))
    synth_train = _load_jsonl(Path("data/processed/v3_1/train.jsonl"))
    synth_val = _load_jsonl(Path("data/processed/v3_1/val.jsonl"))
    synth_unseen = _load_jsonl(Path("data/processed/v3_1/test_unseen_terms.jsonl"))
    llm_80 = _load_jsonl(Path("data/processed/realistic_llm/eval.jsonl"))

    # Create deterministic OCR-perturbed copy of synth_val (6% char noise + accent drop)
    rng = np.random.default_rng(2026)
    synth_val_ocr = [
        augment_document_ocr_noise(d, rng, char_error_rate=0.06, accent_drop_prob=0.6, suffix="ocr_eval")
        for d in synth_val
    ]

    report: dict[str, Any] = {
        "evaluated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pretrained_weights_used": False,
        "self_supervised_artifact": "models/e3c_self_supervised_en_fr.npz",
        "experiments": {},
    }

    # 1. Upgraded CRF on real open-license clinical corpora (with & without E3C self-supervised clusters)
    for name, cfg in (
        ("crf_morph_bigrams", {"use_self_supervised": False, "ocr_augment_ratio": 0.0}),
        ("crf_morph_bigrams_e3c_ss", {"use_self_supervised": True, "ocr_augment_ratio": 0.0}),
        ("crf_morph_bigrams_e3c_ss_ocraug", {"use_self_supervised": True, "ocr_augment_ratio": 0.25}),
    ):
        ts = time.perf_counter()
        m = ClassicalCRFModel(seed=1, cfg=cfg).fit(open_train)
        preds = m.predict_docs(open_test)
        res = ev.evaluate(open_test, preds)
        by_src = {}
        pred_map = {p.doc_id: p for p in preds}
        for src in sorted({d.source for d in open_test}):
            sd = [d for d in open_test if d.source == src]
            sm = ev.evaluate(sd, [pred_map[d.doc_id] for d in sd])
            by_src[src] = {"strict_f1": sm["strict_span"]["f1"], "lenient_f1": sm["lenient_span"]["f1"]}
        report["experiments"][f"open_clinical::{name}"] = {
            "elapsed_s": round(time.perf_counter() - ts, 2),
            "strict_f1": res["strict_span"]["f1"],
            "lenient_f1": res["lenient_span"]["f1"],
            "en_f1": res["per_language"]["en"]["f1"],
            "fr_f1": res["per_language"]["fr"]["f1"],
            "by_corpus": by_src,
        }
        print(
            f"[open::{name:32s}] strict={res['strict_span']['f1']:.4f} lenient={res['lenient_span']['f1']:.4f} "
            f"MC={by_src['maccrobat2020']['strict_f1']:.4f} PX={by_src['pxcorpus']['strict_f1']:.4f} "
            f"E3C={by_src['e3c_layer1']['strict_f1']:.4f}",
            flush=True,
        )

    # 2. OCR-noise robustness & free-text doc-type accuracy on SYNTH-MED-V3.1 -> val, val_ocr, realistic_llm_80
    for name, cfg in (
        ("crf_synth_upgraded", {"use_self_supervised": True, "ocr_augment_ratio": 0.0}),
        ("crf_synth_upgraded_ocraug", {"use_self_supervised": True, "ocr_augment_ratio": 0.30}),
    ):
        ts = time.perf_counter()
        m = ClassicalCRFModel(seed=1, cfg=cfg).fit(synth_train)
        r_val = ev.evaluate(synth_val, m.predict_docs(synth_val))
        r_ocr = ev.evaluate(synth_val_ocr, m.predict_docs(synth_val_ocr))
        r_llm = ev.evaluate(llm_80, m.predict_docs(llm_80))
        report["experiments"][f"synth::{name}"] = {
            "elapsed_s": round(time.perf_counter() - ts, 2),
            "val_clean_strict_f1": r_val["strict_span"]["f1"],
            "val_ocr_strict_f1": r_ocr["strict_span"]["f1"],
            "ocr_drop": round(r_ocr["strict_span"]["f1"] - r_val["strict_span"]["f1"], 4),
            "llm80_strict_f1": r_llm["strict_span"]["f1"],
            "llm80_lenient_f1": r_llm["lenient_span"]["f1"],
            "llm80_doc_acc": r_llm["doc_classification"]["accuracy"],
        }
        print(
            f"[synth::{name:28s}] val_clean={r_val['strict_span']['f1']:.4f} val_ocr={r_ocr['strict_span']['f1']:.4f} "
            f"llm80_f1={r_llm['strict_span']['f1']:.4f} llm80_doc_acc={r_llm['doc_classification']['accuracy']:.4f}",
            flush=True,
        )

    # 3. Neural models: random-init vs from-scratch E3C Layer 3 self-supervised init on unseen_terms
    for model_name, fn, base_cfg in (
        ("bilstm_crf", predict_bilstm_crf, {"hidden_dim": 32, "epochs": 8, "lr": 0.018}),
        ("transformer", predict_transformer, {"hidden_dim": 48, "num_layers": 2, "num_heads": 4, "epochs": 12, "lr": 0.012}),
    ):
        for use_ss in (False, True):
            tag = f"{model_name}_{'e3c_ss' if use_ss else 'rand'}"
            ts = time.perf_counter()
            preds = fn(synth_train, synth_unseen, 1, {**base_cfg, "use_self_supervised": use_ss})
            res = ev.evaluate(synth_unseen, preds)
            report["experiments"][f"unseen_terms::{tag}"] = {
                "elapsed_s": round(time.perf_counter() - ts, 2),
                "strict_f1": res["strict_span"]["f1"],
                "lenient_f1": res["lenient_span"]["f1"],
            }
            print(f"[unseen::{tag:24s}] strict={res['strict_span']['f1']:.4f} lenient={res['lenient_span']['f1']:.4f}", flush=True)

    # 4. Calibrated Stage 3b Assertion Gating check on SYNTH-MED-V3.1 val
    ts = time.perf_counter()
    pipe = HybridPipeline(
        seed=1,
        cfg={
            "hidden_dim": 32,
            "epochs": 6,
            "aux_epochs": 5,
            "lr": 0.018,
            "use_crf_ensemble": True,
            "use_self_supervised": True,
            "use_assertion_bilstm": True,
            "use_transformer_fusion": False,
        },
    ).fit(synth_train)
    p_no_s3b = [pipe.extract_document(d, use_assertion_bilstm=False, use_transformer_fusion=False)[0] for d in synth_val]
    p_with_s3b = [pipe.extract_document(d, use_assertion_bilstm=True, use_transformer_fusion=False)[0] for d in synth_val]
    r_no = ev.evaluate(synth_val, p_no_s3b)
    r_yes = ev.evaluate(synth_val, p_with_s3b)
    report["experiments"]["stage3b_calibrated_gating_val"] = {
        "elapsed_s": round(time.perf_counter() - ts, 2),
        "without_bilstm_assert_macro_f1": r_no["assertion"]["macro_f1"],
        "with_calibrated_bilstm_assert_macro_f1": r_yes["assertion"]["macro_f1"],
        "delta": round(r_yes["assertion"]["macro_f1"] - r_no["assertion"]["macro_f1"], 4),
        "calibrated_class_thresholds": pipe.assertion_model.class_thresholds if pipe.assertion_model else {},
    }
    print(
        f"[stage3b_gating::val] without={r_no['assertion']['macro_f1']:.4f} "
        f"with_calibrated={r_yes['assertion']['macro_f1']:.4f} "
        f"thresholds={report['experiments']['stage3b_calibrated_gating_val']['calibrated_class_thresholds']}",
        flush=True,
    )

    report["total_elapsed_s"] = round(time.perf_counter() - t0, 2)
    out = Path("docs/results/nlp_model_upgrades_benchmark.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out} in {report['total_elapsed_s']}s")


if __name__ == "__main__":
    main()
