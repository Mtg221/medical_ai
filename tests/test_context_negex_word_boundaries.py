"""Step 2 (K3): Word-boundary ConText/NegEx regression tests (EN + FR)."""

from __future__ import annotations

import pytest

from medresearch_platform.approaches.context_negex import (
    CONTEXT_CUES,
    classify_assertion_context,
)


def _classify_substring(sentence: str, target: str, language: str) -> str:
    idx = sentence.index(target)
    return classify_assertion_context(sentence, idx, idx + len(target), language=language)


@pytest.mark.parametrize(
    ("sentence", "target", "language", "expected"),
    [
        # French false-negation regressions ("nie " inside pneumonie / insomnie / neutropénie / hernie)
        ("Le patient a une pneumonie et prend metformine 500 mg.", "metformine", "fr", "present"),
        ("Insomnie et prend zolpidem 10 mg.", "zolpidem", "fr", "present"),
        ("Neutropénie fébrile, traitée par céfépime 2 g.", "céfépime", "fr", "present"),
        ("Hernie inguinale opérée.", "opérée", "fr", "present"),
        # Real French negations + "ni" coordination
        ("Le patient nie toute douleur thoracique.", "douleur thoracique", "fr", "absent"),
        ("Pas de fièvre ni de toux.", "fièvre", "fr", "absent"),
        ("Pas de fièvre ni de toux.", "toux", "fr", "absent"),
        ("Absence d'œdème.", "œdème", "fr", "absent"),
        ("Sans dyspnée.", "dyspnée", "fr", "absent"),
        # English regressions & real negations
        ("Pneumonia and takes metformin.", "metformin", "en", "present"),
        ("insomnia, started zolpidem", "zolpidem", "en", "present"),
        ("denies chest pain", "chest pain", "en", "absent"),
        ("no fever or cough", "fever", "en", "absent"),
        ("no fever or cough", "cough", "en", "absent"),
        ("nontender abdomen", "abdomen", "en", "present"),
        ("Exam of nose shows metformin prescribed.", "metformin", "en", "present"),
        ("Patient has known diabetes and takes metformin.", "metformin", "en", "present"),
        ("Health economy review shows metformin.", "metformin", "en", "present"),
    ],
)
def test_context_negex_k3_word_boundaries(
    sentence: str, target: str, language: str, expected: str
) -> None:
    assert _classify_substring(sentence, target, language) == expected


def _all_cue_cases() -> list[tuple[str, str, str]]:
    cases: list[tuple[str, str, str]] = []
    for lang, groups in CONTEXT_CUES.items():
        for category, cues in groups.items():
            if category in {"terminators", "pseudo_neg"}:
                continue
            for cue in cues:
                cases.append((lang, category, cue))
    return cases


@pytest.mark.parametrize(("lang", "category", "cue"), _all_cue_cases())
def test_every_cue_does_not_fire_inside_another_word(
    lang: str, category: str, cue: str
) -> None:
    core = cue.strip()
    words = core.split()
    variants = [
        " ".join(f"zz{w}" for w in words),
        " ".join(f"{w}zz" for w in words),
        " ".join(f"zz{w}zz" for w in words),
    ]
    for embedded in variants:
        if category == "neg_post":
            sentence = f"Patient has diabetes {embedded}."
        else:
            sentence = f"Patient {embedded} diabetes."
        idx = sentence.index("diabetes")
        got = classify_assertion_context(sentence, idx, idx + len("diabetes"), language=lang)
        assert got == "present", f"Cue {cue!r} ({lang}/{category}) fired inside {embedded!r} -> {got}"
