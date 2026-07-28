from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd

import chem_ts_corr.web as web
import chem_ts_corr.xgb_runner as runner
import chem_ts_corr.xgb_validation as validation
from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.xgb_validation import (
    XGBFeatureSets,
    XGBTimeSplit,
    XGBValidationProvenance,
    XGBValidationResult,
)


def _feature_sets(candidate_count: int = 10) -> XGBFeatureSets:
    rows = 24
    index = pd.date_range("2025-01-01", periods=rows, freq="min")
    target = pd.Series(np.arange(rows, dtype=float) + 1, index=index, name="target")
    data = {
        "target__lag_1": target.to_numpy(),
        "control__lag_1": target.to_numpy() * 0.5,
    }
    candidate_map: dict[str, tuple[str, ...]] = {}
    for number in range(candidate_count):
        variable = f"c{number}"
        feature = f"{variable}__lag_2"
        data[feature] = target.to_numpy() * (number + 2)
        candidate_map[variable] = (feature,)
    features = pd.DataFrame(data, index=index)
    return XGBFeatureSets(
        features=features,
        target=target,
        m0_features=("target__lag_1",),
        m1_features=("target__lag_1", "control__lag_1"),
        m2_features=tuple(features.columns),
        candidate_feature_map=candidate_map,
        max_used_lag=2,
    )


def _splits() -> list[XGBTimeSplit]:
    return [
        XGBTimeSplit(0, slice(0, 8), slice(8, 12), slice(12, 16), 0),
        XGBTimeSplit(1, slice(0, 12), slice(12, 16), slice(16, 24), 0),
    ]


def _pool_with_whitelist() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_order": number + 1,
                "variable": f"c{number}",
                "selection_source": "final_review" if number < 8 else "whitelist",
                "force_included": number >= 8,
            }
            for number in range(10)
        ]
    )


def _pool_with_ten_automatic_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_order": number + 1,
                "variable": f"c{number}",
                "selection_source": "final_review",
                "force_included": False,
            }
            for number in range(10)
        ]
    )


def test_fit_count_is_three_models_plus_every_effective_candidate_per_fold(monkeypatch):
    fit_calls: list[tuple[str, ...]] = []

    class FakeXGBRegressor:
        best_iteration = 1

        def __init__(self, **params):
            self.params = params

        def fit(self, X, y, *, eval_set, verbose):
            fit_calls.append(tuple(X.columns))
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=float)

    monkeypatch.setattr(validation, "XGBRegressor", FakeXGBRegressor)
    feature_sets = _feature_sets()
    splits = _splits()
    pool = _pool_with_whitelist()

    baseline_result = validation.run_xgb_time_validation(feature_sets, splits)
    _, candidate_summary = validation.run_candidate_uplift_validation(
        feature_sets,
        splits,
        pool,
        baseline_result=baseline_result,
    )

    fold_count = len(splits)
    candidate_count = len(candidate_summary)
    assert candidate_count == 10
    assert set(candidate_summary["variable"]) == {f"c{number}" for number in range(10)}
    assert len(fit_calls) == (3 + candidate_count) * fold_count
    assert fit_calls.count(feature_sets.m1_features) == fold_count


def test_uplift_validation_keeps_all_ten_automatic_candidates(monkeypatch):
    class FakeXGBRegressor:
        best_iteration = 1

        def __init__(self, **params):
            self.params = params

        def fit(self, X, y, *, eval_set, verbose):
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=float)

    monkeypatch.setattr(validation, "XGBRegressor", FakeXGBRegressor)
    feature_sets = _feature_sets()
    splits = _splits()
    baseline_result = validation.run_xgb_time_validation(feature_sets, splits)

    _, candidate_summary = validation.run_candidate_uplift_validation(
        feature_sets,
        splits,
        _pool_with_ten_automatic_candidates(),
        baseline_result=baseline_result,
    )

    assert candidate_summary["variable"].tolist() == [f"c{number}" for number in range(10)]


def test_runner_builds_pool_features_and_splits_once(monkeypatch, tmp_path: Path):
    counts = {name: 0 for name in ["pool", "features", "splits", "models", "uplift"]}
    feature_sets = _feature_sets(1)
    splits = [_splits()[0]]
    pool = _pool_with_whitelist().head(1)
    provenance = XGBValidationProvenance(
        m1_features=feature_sets.m1_features,
        split_signature=(),
        parameter_signature=(),
        early_stopping_rounds=50,
        data_fingerprint="performance-contract",
    )
    model_result = XGBValidationResult(
        fold_metrics=pd.DataFrame(),
        summary=pd.DataFrame(),
        predictions=pd.DataFrame(),
        provenance=provenance,
    )
    candidate_summary = pd.DataFrame([{"variable": "c0"}])

    def count(name, result):
        def wrapped(*args, **kwargs):
            counts[name] += 1
            if name == "uplift":
                assert kwargs["baseline_result"] is model_result
            return result

        return wrapped

    monkeypatch.setattr(runner, "XGBRegressor", object)
    monkeypatch.setattr(runner, "build_xgb_candidate_pool", count("pool", pool))
    monkeypatch.setattr(runner, "build_xgb_feature_sets", count("features", feature_sets))
    monkeypatch.setattr(runner, "build_expanding_time_splits", count("splits", splits))
    monkeypatch.setattr(runner, "run_xgb_time_validation", count("models", model_result))
    monkeypatch.setattr(
        runner,
        "run_candidate_uplift_validation",
        count("uplift", (pd.DataFrame(), candidate_summary)),
    )
    data = pd.DataFrame(
        {"target": np.arange(30, dtype=float), "c0": np.arange(30, dtype=float) * 2}
    )
    final = pd.DataFrame(
        [{"variable": "c0", "final_recommendation": "priority_review", "screening_lag": 2}]
    )

    result = runner.run_xgb_validation(
        run_dir=tmp_path,
        target="target",
        data=data,
        final_review_summary=final,
        top_n=1,
        max_lag=2,
    )

    assert result.status == "success"
    assert counts == {name: 1 for name in counts}


def test_web_prepares_xgb_data_once_without_raw_duplicate(monkeypatch, tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    input_path = tmp_path / "input.csv"
    input_path.write_text("timestamp,target,x\n2025-01-01,1,2\n", encoding="utf-8")
    config = AnalysisConfig(input_path, "timestamp", "target", run_dir)
    web._write_run_config(run_dir, config, "file-id")
    pd.DataFrame(
        [{"variable": "x", "final_recommendation": "priority_review", "screening_lag": 2}]
    ).to_csv(run_dir / "final_review_summary.csv", index=False)
    ranked = pd.DataFrame([{"variable": "x", "lag": 2}])
    ranked.to_csv(run_dir / "ranked_features.csv", index=False)
    ranked.to_csv(run_dir / "recommended_candidates.csv", index=False)
    calls = {"prepare": 0, "service": 0}

    def prepare(config):
        calls["prepare"] += 1
        return pd.DataFrame({"target": [1.0], "x": [2.0]})

    def service(**kwargs):
        calls["service"] += 1
        return runner.XGBRunResult("failed", (), None, None, None, "stopped")

    monkeypatch.setattr(web, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(web, "_multipart_form", lambda handler: {
        "run_id": "run", "enable_xgb_validation": "true", "top_n": "1"
    })
    monkeypatch.setattr(web, "_prepared_frame_for_validation", prepare)
    monkeypatch.setattr(web, "run_xgb_analysis", service)

    payload = web._run_xgb_validation_response(object())

    assert payload["status"] == "failed"
    assert calls == {"prepare": 1, "service": 1}
    source = inspect.getsource(web._run_xgb_validation_response)
    assert "load_timeseries_csv" not in source


def test_candidate_loop_has_no_full_frame_deep_copy_or_rebuild():
    source = inspect.getsource(validation.run_candidate_uplift_validation)
    candidate_loop = source.split("for variable, added in valid_features.items():", 1)[1]

    for forbidden in [
        "features.copy(deep=True)",
        "data.copy(deep=True)",
        "candidate_pool.copy(deep=True)",
        "build_xgb_feature_sets",
        "prepare_xgb_validation_frame",
        "build_expanding_time_splits",
    ]:
        assert forbidden not in candidate_loop


def test_xgb_sources_keep_performance_and_ranking_architecture_guards():
    source = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in [
            "chem_ts_corr/xgb_validation.py",
            "chem_ts_corr/xgb_runner.py",
            "chem_ts_corr/web.py",
        ]
    )
    for forbidden in [
        "GridSearchCV",
        "RandomizedSearchCV",
        "Optuna",
        "Bayesian optimization",
        "final_score =",
        "driver_rank =",
        "final_rank =",
    ]:
        assert forbidden not in source
    validation_source = Path("chem_ts_corr/xgb_validation.py").read_text(encoding="utf-8")
    assert "shift(-" not in validation_source
    assert "abs(best_lag)" not in validation_source


def test_xgb_hard_limits_and_benchmark_script_contract():
    assert validation.MAX_XGB_LAG_POINTS == 5000
    assert validation.DEFAULT_XGB_TOP_N == 8
    assert validation.MAX_XGB_AUTO_TOP_N == 10
    assert validation.MAX_XGB_TOTAL_CANDIDATES == 12
    benchmark = Path("scripts/benchmark_xgb_validation.py").read_text(encoding="utf-8")
    for marker in [
        "np.random.default_rng(42)",
        'default=50_000',
        'default=50',
        "DEFAULT_XGB_TOP_N",
        "MAX_XGB_AUTO_TOP_N",
        "candidates must be between 1 and 10",
        'default=360',
        "time.perf_counter()",
        "tracemalloc",
        '"fit_count"',
        'pip install -e ".[xgb]"',
    ]:
        assert marker in benchmark
    assert "between 1 and 8" not in benchmark
    assert "args.candidates <= DEFAULT_XGB_TOP_N" not in benchmark
    validation_source = Path("chem_ts_corr/xgb_validation.py").read_text(encoding="utf-8")
    assert "automatic_count < DEFAULT_XGB_TOP_N" not in validation_source
    assert "automatic_count < MAX_XGB_AUTO_TOP_N" in validation_source
    assert "len(result[\"variable\"].drop_duplicates())" not in validation_source
    assert validation_source.count("_validate_total_xgb_candidate_count(") == 3
    assert "assert elapsed <" not in benchmark


def test_xgb_documentation_describes_all_candidate_limits_and_failure_behavior():
    documentation = Path("docs/xgb_validation.md").read_text(encoding="utf-8")

    for marker in [
        "默认自动候选数量为 8",
        "自动候选数量调整到最多 10",
        "总候选数量最多为 12",
        "超过 12 会返回 `invalid_input`",
        "不会静默截断",
        "`C` 为最终实际候选数量",
    ]:
        assert marker in documentation
    assert "默认最多选择 8 个自动候选，白名单候选可在自动候选之外强制加入" not in documentation


def test_xgb_documentation_has_required_sections_and_readme_link():
    documentation = Path("docs/xgb_validation.md").read_text(encoding="utf-8")
    for heading in [
        "## 1. 目的",
        "## 2. 前置条件",
        "## 3. 模型定义",
        "## 4. 数据口径",
        "## 5. 时间切分",
        "## 6. 状态解释",
        "## 7. 输出文件",
        "## 8. 性能说明",
        "## 9. 故障排查",
        "## 10. 解释边界",
    ]:
        assert heading in documentation
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "[XGB 四级验证说明](docs/xgb_validation.md)" in readme
