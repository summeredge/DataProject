from chem_ts_corr import web


def test_review_pool_ui_is_distinct_from_initial_screening_candidates():
    assert "二级验证复核池" in web.INDEX_HTML
    assert "复核池独立于一级初筛候选池" in web.INDEX_HTML
    assert "加入复核池" in web.INDEX_HTML
    assert "verificationReviewPool" in web.INDEX_HTML
    assert "/api/add_to_verification_review_pool" in web.INDEX_HTML


def test_review_pool_restores_model_discovery_choices_from_result_payload():
    render = web.INDEX_HTML.split("function renderAnalysisResult(data)", 1)[1].split(
        "function renderPendingBranchResult", 1
    )[0]
    assert "lastModelDiscoveredRows = data.modelDiscoveredCandidates || [];" in render
    assert "syncModelDiscoveryReviewPoolOptions(lastModelDiscoveredRows);" in render
