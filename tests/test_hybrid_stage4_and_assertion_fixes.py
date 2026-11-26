from __future__ import annotations

import pytest

from medresearch_platform.approaches.context_negex import classify_assertion_context
from medresearch_platform.hybrid.stage4_temporal import normalize_timex


@pytest.mark.parametrize(
    ("text", "ttype", "doc_date", "expected"),
    [
        ("2025-03-14", "DATE", "2025-04-01", "2025-03-14"),
        ("14/03/2025", "DATE", "2025-04-01", "2025-03-14"),
        ("March 14, 2025", "DATE", "2025-04-01", "2025-03-14"),
        ("14 mars 2025", "DATE", "2025-04-01", "2025-03-14"),
        ("March 14th", "DATE", "2025-04-01", "2025-03-14"),
        ("14 mars", "DATE", "2025-04-01", "2025-03-14"),
        ("March 2025", "DATE", "2025-04-01", "2025-03"),
        ("mars 2025", "DATE", "2025-04-01", "2025-03"),
        ("yesterday", "RELATIVE", "2025-04-01", "2025-03-31"),
        ("la veille", "RELATIVE", "2025-04-01", "2025-03-31"),
        ("today", "RELATIVE", "2025-04-01", "2025-04-01"),
        ("aujourd'hui", "RELATIVE", "2025-04-01", "2025-04-01"),
        ("3 days ago", "RELATIVE", "2025-04-01", "2025-03-29"),
        ("il y a 3 jours", "RELATIVE", "2025-04-01", "2025-03-29"),
        ("2 weeks ago", "RELATIVE", "2025-04-01", "2025-03-18"),
        ("depuis 2 semaines", "RELATIVE", "2025-04-01", "2025-03-18"),
        ("in 2 weeks", "RELATIVE", "2025-04-01", "2025-04-15"),
        ("dans 2 semaines", "RELATIVE", "2025-04-01", "2025-04-15"),
        ("1 month ago", "RELATIVE", "2025-07-05", "2025-06-05"),
        ("il y a 1 mois", "RELATIVE", "2025-11-14", "2025-10-15"),
        ("POD 3", "RELATIVE", "2025-04-10", "2025-04-10"),
        ("J3 post-op", "RELATIVE", "2025-04-10", "2025-04-10"),
        ("for 7 days", "DURATION", "2025-04-01", "P7D"),
        ("pendant 7 jours", "DURATION", "2025-04-01", "P7D"),
        ("for 2 weeks", "DURATION", "2025-04-01", "P2W"),
        ("pendant 3 mois", "DURATION", "2025-04-01", "P3M"),
        ("once daily", "FREQUENCY", "2025-04-01", "QD"),
        ("twice daily", "FREQUENCY", "2025-04-01", "BID"),
        ("b.i.d.", "FREQUENCY", "2025-04-01", "BID"),
        ("as needed", "FREQUENCY", "2025-04-01", "PRN"),
    ],
)
def test_normalize_timex_en_fr_formats(
    text: str, ttype: str, doc_date: str, expected: str
) -> None:
    assert normalize_timex(text, ttype, doc_date) == expected


@pytest.mark.parametrize(
    ("sentence", "target", "lang", "expected_assertion"),
    [
        ("Douleur thoracique à confirmer lors du suivi.", "Douleur thoracique", "fr", "possible"),
        ("Mère diabétique de type 2.", "diabétique", "fr", "family"),
        ("No evidence of pneumonia on chest X-ray.", "pneumonia", "en", "absent"),
        ("Rule out pulmonary embolism.", "pulmonary embolism", "en", "possible"),
        ("Patient denies fever or chills.", "fever", "en", "absent"),
    ],
)
def test_context_cues_cover_v3_1_minority_classes(
    sentence: str, target: str, lang: str, expected_assertion: str
) -> None:
    start = sentence.index(target)
    end = start + len(target)
    assert classify_assertion_context(sentence, start, end, language=lang) == expected_assertion
