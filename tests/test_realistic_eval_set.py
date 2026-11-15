"""Tests for the 80-note (40 EN + 40 FR) realistic clinical prose evaluation set."""

from __future__ import annotations

from medresearch_platform.data.realistic_v1 import generate_realistic_eval_set


def test_realistic_eval_set_offsets_and_balance() -> None:
    docs = generate_realistic_eval_set(n_en=40, n_fr=40, seed=2026)
    assert len(docs) == 80
    assert sum(1 for d in docs if d.language == "en") == 40
    assert sum(1 for d in docs if d.language == "fr") == 40

    for d in docs:
        assert len(d.entities) >= 5
        assert len(d.timex) >= 2
        assert len(d.relations) >= 2
        for ent in d.entities:
            assert d.text[ent.start : ent.end] == ent.text
        for tx in d.timex:
            assert d.text[tx.start : tx.end] == tx.text
