# Reference Review Requests (Human-in-the-Loop — Constraint C7)

Per Constraint C7, `tests/fixtures/noise_0027_ref_draft.json` and `tests/fixtures/noise_0038_ref_draft.json` were **NOT edited** and remain `"status": "draft_unreviewed"`. The following proposals are submitted for human clinical/owner review:

## 1. `n27_07` (`tests/fixtures/noise_0027_ref_draft.json`)
- **Current entry:** `{"id": "n27_07", "category": "return_precautions", "text": "worsening headache", "context": "seek urgent evaluation", "expected_assertion_detail": "conditional"}`
- **Evidence:** In `tests/fixtures/noise_0027.txt` (line 103), `"worsening headache"` precedes `"seek urgent evaluation"` by 52 characters on the same line (`- Return precautions text partially corrupted: ‘worsening headache, weakness, speech / vision changes—seek urgent evaluation’ [fragment].`). Evaluators that search forward from `context` will miss `text` unless they search the enclosing line.
- **Suggested change:** Update `context` to `"Return precautions text partially corrupted: ‘worsening headache"` (or add explicit character offsets `[start, end]`) during formal human annotation review.

## 2. General Sign-Off Request
- Both `noise_0027_ref_draft.json` and `noise_0038_ref_draft.json` require full span-level clinical annotation and sign-off (`reviewer_signoff`, `reviewed_at`) before being promoted from `draft_unreviewed` to human-reviewed gold benchmarks.
