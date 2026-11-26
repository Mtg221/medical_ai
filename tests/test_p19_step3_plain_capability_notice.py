"""Prompt 19 — Step 3: Verify identical plain EN/FR served capability summary in README, model card, and UI."""

from __future__ import annotations

import json
from pathlib import Path


def test_p19_step3_plain_capability_notice_matches_served_eval() -> None:
    served = json.loads(Path("docs/results/vision/p18_step1_served_path_eval.json").read_text(encoding="utf-8"))
    derm_auroc = f"{served['served_path_by_modality']['dermoscopic']['auroc']:.4f}"
    clin_auroc = f"{served['served_path_by_modality']['clinical']['auroc']:.4f}"
    tbp_auroc = f"{served['served_path_by_modality']['tbp_tile']['auroc']:.4f}"
    reweighted = served["view_b_corpus_prevalence_reweighted"]["overall"]
    abs_pct = f"{float(reweighted['abstention_rate']) * 100:.2f}%"
    ece_val = f"{float(reweighted['ece']):.4f}"

    en_sentence = (
        f"In practice, the image tool only gives a risk band for dermoscopic photos (served AUROC {derm_auroc}); "
        f"clinical phone photos get no risk band (AUROC {clin_auroc}), TBP tiles abstain (AUROC {tbp_auroc}), "
        f"and at corpus prevalence served abstention is {abs_pct} (ECE {ece_val})."
    )
    fr_sentence = (
        f"En pratique, l'outil d'imagerie ne fournit une catégorie de risque que pour les photos dermoscopiques "
        f"(AUROC servie {derm_auroc}) ; les photos cliniques ne reçoivent pas de catégorie de risque "
        f"(AUROC {clin_auroc}), les tuiles TBP s'abstiennent (AUROC {tbp_auroc}) et, à la prévalence du corpus, "
        f"l'abstention servie est de {abs_pct} (ECE {ece_val})."
    )

    for doc_path in (
        Path("README.md"),
        Path("docs/model-cards/MODEL_CARD_DERM_CNN_V1.md"),
        Path("frontend/src/i18n/index.ts"),
    ):
        text = doc_path.read_text(encoding="utf-8")
        assert en_sentence in text, f"Missing EN served capability sentence in {doc_path}"
        assert fr_sentence in text, f"Missing FR served capability sentence in {doc_path}"

    upload_page = Path("frontend/src/pages/UploadJobsPage.tsx").read_text(encoding="utf-8")
    assert 't("upload.servedCapabilityNotice")' in upload_page
