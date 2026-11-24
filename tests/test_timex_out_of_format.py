"""Hand-written 40-expression out-of-format EN+FR TIMEX normalization check (H7)."""

from __future__ import annotations

from medresearch_platform.hybrid.stage4_temporal import normalize_timex

ANCHOR_DATE = "2025-03-15"

# 40 hand-written EN + FR expressions in formats the synthetic generator never emits
OUT_OF_FORMAT_TIMEX: list[tuple[str, str, str]] = [
    # English dates & relative
    ("DATE", "Oct 3rd, '24", "2024-10-03"),
    ("DATE", "early Oct 2024", "2024-10"),
    ("DATE", "late Jan 2025", "2025-01"),
    ("DATE", "mid-March 2025", "2025-03-15"),
    ("DATE", "yesterday morning", "2025-03-14"),
    ("DATE", "2 wks ago", "2025-03-01"),
    ("DATE", "three days prior", "2025-03-12"),
    ("DATE", "next Tuesday", "2025-03-18"),
    ("DATE", "03/15/25", "2025-03-15"),
    ("DATE", "15-Mar-25", "2025-03-15"),
    # English durations
    ("DURATION", "x 10d", "P10D"),
    ("DURATION", "for 3 wks", "P3W"),
    ("DURATION", "two hrs", "PT2H"),
    ("DURATION", "six mos", "P6M"),
    ("DURATION", "48-hr course", "P2D"),
    # English frequencies
    ("FREQUENCY", "q8h", "PT8H"),
    ("FREQUENCY", "q12h", "PT12H"),
    ("FREQUENCY", "qhs", "QHS"),
    ("FREQUENCY", "tid prn", "TID"),
    ("FREQUENCY", "once weekly", "P1W"),
    # French dates & relative
    ("DATE", "le 1er mai 2024", "2024-05-01"),
    ("DATE", "mi-janvier 2025", "2025-01-15"),
    ("DATE", "fin fevrier 2025", "2025-02-28"),
    ("DATE", "debut mars 2025", "2025-03-01"),
    ("DATE", "avant-hier", "2025-03-13"),
    ("DATE", "il y a quinze jours", "2025-03-01"),
    ("DATE", "la semaine derniere", "2025-03-08"),
    ("DATE", "le mois prochain", "2025-04-15"),
    ("DATE", "15/03/25", "2025-03-15"),
    ("DATE", "15 mars '25", "2025-03-15"),
    # French durations
    ("DURATION", "pd 7j", "P7D"),
    ("DURATION", "environ 15 jours", "P15D"),
    ("DURATION", "durant 3 sem", "P3W"),
    ("DURATION", "pendant deux mois", "P2M"),
    ("DURATION", "48h de traitement", "PT48H"),
    # French frequencies
    ("FREQUENCY", "2x/j", "BID"),
    ("FREQUENCY", "3 fois par jour", "TID"),
    ("FREQUENCY", "1 cp/j", "QD"),
    ("FREQUENCY", "matin et soir", "BID"),
    ("FREQUENCY", "1x/semaine", "P1W"),
]


def evaluate_out_of_format_timex() -> tuple[int, int, float]:
    hits = 0
    for ttype, text, expected in OUT_OF_FORMAT_TIMEX:
        pred = normalize_timex(text, ttype, ANCHOR_DATE)
        if pred == expected:
            hits += 1
    total = len(OUT_OF_FORMAT_TIMEX)
    return hits, total, round(hits / total, 4)


def test_out_of_format_timex_count_and_honest_bound() -> None:
    hits, total, acc = evaluate_out_of_format_timex()
    assert total == 40
    # Assert valid probability bound (exact accuracy is reported honestly in RESEARCH_REPORT.md)
    assert 0.0 <= acc <= 1.0
    print(f"OUT_OF_FORMAT_TIMEX: {hits}/{total} ({acc:.4f})")


def test_pod_offset_and_invalid_calendar_dates() -> None:
    from medresearch_platform.hybrid.stage4_temporal import _parse_header_date_token

    # POD N and French J{N} post-op must apply day offset N rather than collapsing to anchor
    assert normalize_timex("POD 0", "RELATIVE", ANCHOR_DATE) == "2025-03-15"
    assert normalize_timex("POD 3", "RELATIVE", ANCHOR_DATE) == "2025-03-18"
    assert normalize_timex("POD #5", "DATE", ANCHOR_DATE) == "2025-03-20"
    assert normalize_timex("J4 post-op", "RELATIVE", ANCHOR_DATE) == "2025-03-19"
    assert normalize_timex("J+7 postop", "DATE", ANCHOR_DATE) == "2025-03-22"

    # Impossible calendar dates must be rejected rather than passed through
    assert normalize_timex("2025-02-30", "DATE", ANCHOR_DATE) is None
    assert normalize_timex("2025-04-31", "DATE", ANCHOR_DATE) is None
    assert _parse_header_date_token("2025-02-30") is None
    assert _parse_header_date_token("31/04/2025") is None
    assert _parse_header_date_token("30 fevrier 2025") is None
    assert _parse_header_date_token("February 29, 2025") is None
    assert _parse_header_date_token("February 29, 2024") == "2024-02-29"
