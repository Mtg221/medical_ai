"""Patient/document-level stratified splitting and MinHash near-duplicate leakage checks."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Any

from medresearch_platform.data.schema import AnnotatedDocument

_MERSENNE_PRIME = (1 << 61) - 1


def _shingles(text: str, k: int = 5) -> set[str]:
    norm = " ".join(text.lower().split())
    if len(norm) <= k:
        return {norm} if norm else set()
    return {norm[i : i + k] for i in range(len(norm) - k + 1)}


def compute_minhash_signature(text: str, *, num_perm: int = 64, shingle_size: int = 5, seed: int = 42) -> tuple[int, ...]:
    """Compute a deterministic MinHash signature over character k-shingles."""
    shingles = _shingles(text, k=shingle_size)
    if not shingles:
        return tuple(0 for _ in range(num_perm))
    rng = random.Random(seed)
    coeffs = [
        (rng.randrange(1, _MERSENNE_PRIME), rng.randrange(0, _MERSENNE_PRIME))
        for _ in range(num_perm)
    ]
    base_hashes = [
        int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")
        for s in shingles
    ]
    sig: list[int] = []
    for a, b in coeffs:
        min_val = min((a * h + b) % _MERSENNE_PRIME for h in base_hashes)
        sig.append(min_val)
    return tuple(sig)


def minhash_similarity(sig_a: tuple[int, ...], sig_b: tuple[int, ...]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if not sig_a or len(sig_a) != len(sig_b):
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b, strict=True) if a == b)
    return matches / len(sig_a)


def assign_stratified_splits(
    docs: list[AnnotatedDocument],
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> list[AnnotatedDocument]:
    """
    Assign train/val/test splits at the patient_id level, stratified by (language, doc_type).
    All documents sharing a patient_id are placed in the same split.
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    patient_groups: dict[str, list[AnnotatedDocument]] = defaultdict(list)
    for d in docs:
        patient_groups[d.patient_id].append(d)

    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for pid, pdocs in patient_groups.items():
        primary = pdocs[0]
        strata[(primary.language, primary.doc_type)].append(pid)

    rng = random.Random(seed)
    pid_to_split: dict[str, str] = {}

    for _stratum, pids in sorted(strata.items()):
        shuffled = list(pids)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_test = max(1, round(n * test_ratio)) if n >= 3 else 0
        n_val = max(1, round(n * val_ratio)) if n >= 3 else 0
        n_train = max(1, n - n_test - n_val)
        while n_train + n_val + n_test > n and n_val > 0:
            n_val -= 1
        while n_train + n_val + n_test > n and n_test > 0:
            n_test -= 1

        for pid in shuffled[:n_train]:
            pid_to_split[pid] = "train"
        for pid in shuffled[n_train : n_train + n_val]:
            pid_to_split[pid] = "val"
        for pid in shuffled[n_train + n_val :]:
            pid_to_split[pid] = "test"

    for d in docs:
        d.split = pid_to_split[d.patient_id]
    return docs


def verify_no_split_leakage(
    docs: list[AnnotatedDocument],
    *,
    minhash_threshold: float = 0.85,
    num_perm: int = 64,
    shingle_size: int = 5,
) -> dict[str, Any]:
    """Verify zero patient_id overlap and zero MinHash near-duplicate text across splits."""
    by_split: dict[str, list[AnnotatedDocument]] = defaultdict(list)
    for d in docs:
        if d.split is None:
            raise ValueError(f"Document {d.doc_id} has no split assigned")
        by_split[d.split].append(d)

    patient_sets = {sp: {d.patient_id for d in sdocs} for sp, sdocs in by_split.items()}
    shared_patients: list[str] = []
    split_names = sorted(by_split.keys())
    for i, sp_a in enumerate(split_names):
        for sp_b in split_names[i + 1 :]:
            overlap = sorted(patient_sets[sp_a] & patient_sets[sp_b])
            shared_patients.extend(f"{sp_a}<->{sp_b}:{pid}" for pid in overlap)

    sigs: dict[str, tuple[int, ...]] = {
        d.doc_id: compute_minhash_signature(d.text, num_perm=num_perm, shingle_size=shingle_size)
        for d in docs
    }

    near_duplicates: list[dict[str, Any]] = []
    for i, sp_a in enumerate(split_names):
        for sp_b in split_names[i + 1 :]:
            for da in by_split[sp_a]:
                sig_a = sigs[da.doc_id]
                for db in by_split[sp_b]:
                    sim = minhash_similarity(sig_a, sigs[db.doc_id])
                    if sim >= minhash_threshold:
                        near_duplicates.append(
                            {
                                "split_a": sp_a,
                                "doc_a": da.doc_id,
                                "split_b": sp_b,
                                "doc_b": db.doc_id,
                                "similarity": round(sim, 4),
                            }
                        )

    passed = (len(shared_patients) == 0) and (len(near_duplicates) == 0)
    return {
        "passed": passed,
        "shared_patients": shared_patients,
        "near_duplicates": near_duplicates,
        "minhash_threshold": minhash_threshold,
        "split_counts": {sp: len(sdocs) for sp, sdocs in by_split.items()},
    }
