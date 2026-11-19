"""Prompt 19 — Step 1: Verify frozen SYNTH-MED-V3.1 val/test sets, hashes, eval paths, and train isolation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from medresearch_platform.data.synthetic_v3 import generate_v3_corpus_splits

LOCKED_DIR = Path("data/locked/synth_med_v3_1")
LOCKED_SPLITS = ("val", "test_iid", "test_unseen_terms", "test_unseen_templates")


def test_locked_synth_med_v3_1_hashes_match_manifest_and_generator() -> None:
    manifest = json.loads((LOCKED_DIR / "LOCK_MANIFEST.json").read_text(encoding="utf-8"))
    expected_hashes = manifest["sha256"]
    splits = generate_v3_corpus_splits(seed=42, version="v3_1")

    for sp in LOCKED_SPLITS:
        locked_file = LOCKED_DIR / f"{sp}.jsonl"
        assert locked_file.exists(), f"Missing locked split file: {locked_file}"
        actual_sha = hashlib.sha256(locked_file.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        assert actual_sha == expected_hashes[sp], f"Locked file tampered: {sp}"

        gen_bytes = ("\n".join(json.dumps(d.to_dict(), ensure_ascii=False) for d in splits[sp]) + "\n").encode("utf-8")
        gen_sha = hashlib.sha256(gen_bytes).hexdigest()
        assert gen_sha == expected_hashes[sp], f"Generator drifted for locked split '{sp}': {gen_sha} != {expected_hashes[sp]}"


def test_eval_scripts_read_from_locked_dir() -> None:
    for script_path in (
        Path("scripts/run_p18_step6_locked_eval.py"),
        Path("scripts/run_step6_w9_text_reeval.py"),
    ):
        txt = script_path.read_text(encoding="utf-8")
        assert "data/locked/synth_med_v3_1" in txt, f"{script_path} must read from data/locked/synth_med_v3_1"


def test_train_has_zero_locked_doc_texts_or_unseen_terms() -> None:
    splits = generate_v3_corpus_splits(seed=42, version="v3_1")
    train_docs = splits["train"]
    train_texts = {d.text.strip() for d in train_docs}

    locked_texts: set[str] = set()
    for sp in LOCKED_SPLITS:
        for ln in (LOCKED_DIR / f"{sp}.jsonl").read_text(encoding="utf-8").splitlines():
            if ln.strip():
                locked_texts.add(str(json.loads(ln)["text"]).strip())
    assert not (train_texts & locked_texts), "Train split overlaps locked val/test document texts!"

    unseen_types = {"ANATOMY", "DIAGNOSIS", "LAB_TEST", "MEDICATION", "PROCEDURE", "SYMPTOM"}
    train_terms_by_type: dict[str, set[str]] = {t: set() for t in unseen_types}
    for d in train_docs:
        for e in d.entities:
            if e.type in unseen_types:
                train_terms_by_type[e.type].add(e.text.lower().strip())

    unseen_lines = [ln for ln in (LOCKED_DIR / "test_unseen_terms.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    for ln in unseen_lines:
        doc_obj = json.loads(ln)
        for e in doc_obj.get("entities", []):
            et = str(e.get("type") or "")
            if et in unseen_types:
                surf = str(e.get("text") or "").lower().strip()
                assert surf not in train_terms_by_type[et], f"Unseen term {surf!r} ({et}) leaked into train!"
