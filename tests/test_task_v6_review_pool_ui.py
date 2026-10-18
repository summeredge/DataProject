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


def test_residual_candidates_offer_run_bound_review_pool_actions():
    renderer = web.INDEX_HTML.split("function renderCompactDetailTable(", 1)[1].split(
        "function selectCompactDetailRow", 1
    )[0]
    assert "targetId === recommendedCandidateTable" in renderer
    assert "lastVerificationReviewPoolRows" in renderer
    assert '"已在复核池" : "加入复核池"' in renderer
    assert "alreadyInPool || !runId || isRestoredRunReadOnly()" in renderer
    assert 'addToVerificationReviewPool("manual_include", row.variable, runId)' in renderer

    add = web.INDEX_HTML.split("async function addToVerificationReviewPool(", 1)[1].split(
        "function includesFlag", 1
    )[0]
    assert 'form.append("run_id", runId);' in add
    assert "if (currentRunId !== runId) return;" in add
    assert "renderRecommendedCandidateTable(lastRecommendedRows);" in add
