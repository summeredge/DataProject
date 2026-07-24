from __future__ import annotations

from chem_ts_corr import web


def test_result_page_has_no_manual_reordering_controls_or_api():
    source = web.INDEX_HTML
    forbidden = [
        "candidateDecisionControls",
        "applyCandidateDecision",
        "人工推荐决策",
        "保存并重排",
        "/api/update_candidate_decision",
    ]

    assert all(token not in source for token in forbidden)
    assert not hasattr(web, "_update_candidate_decision_response")
    assert "candidate_decision_records.json" not in web.DOWNLOAD_FILES
    assert "reordered_recommendations.csv" not in web.DOWNLOAD_FILES
    assert "recommended_candidates_reordered.csv" not in web.DOWNLOAD_FILES
