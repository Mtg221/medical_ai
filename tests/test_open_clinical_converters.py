"""Offline unit tests for MACCROBAT2020, PxCorpus, and E3C Layer 1 corpus converters."""

from __future__ import annotations

from medresearch_platform.data.converters import (
    from_e3c_xmi,
    from_maccrobat_brat,
    from_pxcorpus_row,
)


def test_from_maccrobat_brat_maps_entities_relations_and_timex() -> None:
    text = "Patient had severe fever and pneumonia; started amoxicillin 500 mg daily on 2020-03-10."
    ann = "\n".join(
        [
            "T1\tSign_symptom 19 24\tfever",
            "T2\tDisease_disorder 29 38\tpneumonia",
            "T3\tMedication 48 59\tamoxicillin",
            "T4\tDosage 60 66\t500 mg",
            "T5\tFrequency 67 72\tdaily",
            "T6\tDate 76 86\t2020-03-10",
            "R1\tCAUSE Arg1:T2 Arg2:T1",
        ]
    )
    doc = from_maccrobat_brat(text, ann, doc_id="maccrobat_test_1")
    assert doc.language == "en"
    assert doc.metadata["license"] == "CC-BY-4.0"
    assert [(e.type, e.text) for e in doc.entities] == [
        ("SYMPTOM", "fever"),
        ("DIAGNOSIS", "pneumonia"),
        ("MEDICATION", "amoxicillin"),
        ("DOSAGE", "500 mg"),
        ("FREQUENCY", "daily"),
        ("DATE", "2020-03-10"),
    ]
    assert len(doc.relations) == 1
    assert doc.relations[0].type == "CAUSES"
    assert {t.type for t in doc.timex} == {"DATE", "FREQUENCY"}
    assert doc.validate_spans() == []


def test_from_pxcorpus_row_merges_dosage_and_duration_spans() -> None:
    tag_names = [
        "O",
        "B-drug",
        "B-d_dos_val",
        "B-d_dos_up",
        "B-roa",
        "B-dur_val",
        "B-dur_ut",
        "B-rhythm_tdte",
    ]
    row = {
        "id": "42",
        "tokens": ["flagyl", "500", "milligrammes", "oral", "matin", "pendant", "8", "jours"],
        "ner_tags": [1, 2, 3, 4, 7, 0, 5, 6],
    }
    doc = from_pxcorpus_row(row, tag_names, split="train")
    assert doc is not None
    assert doc.language == "fr"
    assert doc.doc_type == "prescription"
    assert doc.metadata["license"] == "CC-BY-4.0"
    assert [(e.type, e.text) for e in doc.entities] == [
        ("MEDICATION", "flagyl"),
        ("DOSAGE", "500 milligrammes"),
        ("ROUTE", "oral"),
        ("FREQUENCY", "matin"),
        ("DURATION", "8 jours"),
    ]
    assert any(t.type == "DURATION" and t.normalized == "P8D" for t in doc.timex)
    assert doc.validate_spans() == []


def test_from_e3c_xmi_extracts_entities_negation_and_timex() -> None:
    sofa = "Homme de 55 ans sans fièvre avec diabète au cou depuis deux mois (110/70 mmHg)."
    xmi = f"""<?xml version="1.0" encoding="UTF-8"?>
<xmi:XMI xmlns:xmi="http://www.omg.org/XMI" xmlns:cas="http:///uima/cas.ecore" xmlns:custom="http:///webanno/custom.ecore">
  <cas:Sofa xmi:id="1" sofaString="{sofa}" />
  <custom:METADATA xmi:id="2" docTime="2019-05-01" docLicense="CC BY 4.0" docSource="PAMJ" docDOI="10.1/test" />
  <custom:EVENT xmi:id="3" begin="21" end="27" polarity="NEG" contextualModality="ACTUAL" />
  <custom:CLINENTITY xmi:id="4" begin="21" end="27" entityID="C0015967" />
  <custom:CLINENTITY xmi:id="5" begin="33" end="40" entityID="C0011860" />
  <custom:BODYPART xmi:id="6" begin="44" end="47" />
  <custom:TIMEX3 xmi:id="7" begin="55" end="64" timex3Class="DURATION" value="P2M" />
  <custom:RML xmi:id="8" begin="66" end="77" />
</xmi:XMI>"""
    doc = from_e3c_xmi(xmi, doc_id="e3c_fr_test", language="fr")
    assert doc is not None
    assert doc.language == "fr"
    assert doc.document_date == "2019-05-01"
    assert doc.metadata["license"] == "CC BY 4.0"
    assert [(e.type, e.text, e.assertion) for e in doc.entities] == [
        ("DIAGNOSIS", "fièvre", "absent"),
        ("DIAGNOSIS", "diabète", "present"),
        ("ANATOMY", "cou", "present"),
        ("DURATION", "deux mois", "present"),
        ("LAB_VALUE", "110/70 mmHg", "present"),
    ]
    assert len(doc.timex) == 1
    assert doc.timex[0].normalized == "P2M"
    assert doc.validate_spans() == []
