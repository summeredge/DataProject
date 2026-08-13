from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

import chem_ts_corr.cli as cli
import chem_ts_corr.pipeline as pipeline
from chem_ts_corr.config import AnalysisConfig


FORMAL_MODES = ["raw", "lowpass", "lowpass_detrend", "lowpass_diff"]


def _analyze_parser() -> argparse.ArgumentParser:
    subparsers = [
        action
        for action in cli.build_parser()._actions
        if isinstance(action, argparse._SubParsersAction)
    ][0]
    return subparsers.choices["analyze"]


def _write_input(tmp_path: Path, mode: str = "raw") -> Path:
    rows = 120
    time = np.arange(rows, dtype=float)
    frame = pd.DataFrame(
        {
            "target": np.sin(time / 7),
            "candidate": np.sin((time + 2) / 7),
            "control": np.cos(time / 9),
        },
        index=pd.date_range("2026-01-01", periods=rows, freq="min"),
    )
    input_path = tmp_path / "input.csv"
    table = frame.copy()
    table["time"] = frame.index
    table[["time", *frame.columns]].to_csv(
        input_path, index=False, encoding="utf-8-sig"
    )
    return input_path


def _cli_args(command: list[str]) -> list[str]:
    return ["chem-ts-corr", *command]


# --- 1. analyze choices only contain the four formal modes -----------------


def test_analyze_preprocess_mode_choices_are_only_formal_modes():
    analyze = _analyze_parser()
    mode_action = next(action for action in analyze._actions if action.dest == "preprocess_mode")

    assert sorted(mode_action.choices) == sorted(FORMAL_MODES)
    assert mode_action.default == "raw"


def test_analyze_rejects_legacy_mode_choices():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "analyze",
                "--input", "a.csv",
                "--time-column", "t",
                "--target", "y",
                "--output", "out",
                "--preprocess-mode", "detrend",
            ]
        )


# --- 2/3. tau and diff parameters ------------------------------------------


def test_analyze_parser_exposes_tau_and_diff_defaults():
    analyze = _analyze_parser()

    assert analyze.get_default("lowpass_tau_minutes") == 5.0
    assert analyze.get_default("diff_interval_minutes") is None
    assert analyze.get_default("detrend_window") == 24


def test_analyze_parser_passes_tau_and_diff_values():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "analyze",
            "--input", "a.csv",
            "--time-column", "t",
            "--target", "y",
            "--output", "out",
            "--preprocess-mode", "lowpass_diff",
            "--lowpass-tau-minutes", "7.5",
            "--diff-interval-minutes", "5",
        ]
    )

    assert args.lowpass_tau_minutes == 7.5
    assert args.diff_interval_minutes == 5.0


# --- 4/5. analyze uses the unified workflow --------------------------------


def test_analyze_raw_calls_unified_workflow(monkeypatch, tmp_path):
    input_path = _write_input(tmp_path)
    captured: dict[str, object] = {}

    def fake_workflow(config):
        captured["config"] = config
        return {"branch": "raw", "timings": {}}

    monkeypatch.setattr(
        cli,
        "run_initial_screening_workflow",
        fake_workflow,
    )
    monkeypatch.setattr(sys, "argv", _cli_args([
        "analyze",
        "--input", str(input_path),
        "--time-column", "time",
        "--target", "target",
        "--output", str(tmp_path / "out"),
        "--preprocess-mode", "raw",
    ]))

    cli.main()

    config = captured["config"]
    assert isinstance(config, AnalysisConfig)
    assert config.preprocess_mode == "raw"
    assert config.lowpass_tau_minutes == 5.0
    assert config.diff_interval_minutes is None
    assert (tmp_path / "out" / "run_config.json").exists()
    assert not hasattr(cli, "run_analysis")


@pytest.mark.parametrize("mode", ["lowpass", "lowpass_detrend", "lowpass_diff"])
def test_analyze_processed_does_not_auto_confirm(monkeypatch, tmp_path, mode: str):
    input_path = _write_input(tmp_path)
    confirm_calls: list[tuple[object, ...]] = []
    workflow_calls: list[AnalysisConfig] = []
    monkeypatch.setattr(
        cli,
        "run_initial_screening_workflow",
        lambda config: (
            workflow_calls.append(config)
            or {
                "raw": {},
                "processed": {},
                "comparison_path": Path(config.output_dir) / "preprocessing_comparison.csv",
                "context_path": Path(config.output_dir) / "preprocessing_context.json",
            }
        ),
    )
    monkeypatch.setattr(
        cli,
        "confirm_initial_screening_branch",
        lambda output_dir, branch: confirm_calls.append((output_dir, branch)),
    )
    monkeypatch.setattr(sys, "argv", _cli_args([
        "analyze",
        "--input", str(input_path),
        "--time-column", "time",
        "--target", "target",
        "--output", str(tmp_path / "out"),
        "--preprocess-mode", mode,
    ]))

    cli.main()

    assert len(workflow_calls) == 1
    assert workflow_calls[0].preprocess_mode == mode
    assert confirm_calls == []


# --- 6/7. confirm-branch ---------------------------------------------------


@pytest.mark.parametrize("branch", ["raw", "processed"])
def test_confirm_branch_calls_backend_confirmation(monkeypatch, tmp_path, branch: str):
    captured: list[tuple[object, str]] = []
    monkeypatch.setattr(
        cli,
        "confirm_initial_screening_branch",
        lambda output_dir, branch: captured.append((output_dir, branch)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        _cli_args(["confirm-branch", "--output", str(tmp_path), "--branch", branch]),
    )

    cli.main()

    assert captured == [(tmp_path, branch)]


# --- 8-12. run-* downstream commands ---------------------------------------


@pytest.mark.parametrize(
    ("command", "runner_name", "extra_args"),
    [
        ("run-enhanced", "run_enhanced_screening_for_active_branch", []),
        ("run-granger", "run_granger_for_active_branch", []),
        ("run-model", "run_model_for_active_branch", []),
        (
            "run-causal-review",
            "run_causal_review_for_active_branch",
            ["--control-columns", "c1,c2", "--maxlag", "5", "--min-rows", "60", "--top-n", "4"],
        ),
        (
            "run-xgb",
            "run_xgb_for_active_branch",
            ["--control-columns", "c1", "--whitelist", "w1,w2", "--top-n", "6", "--max-lag", "12"],
        ),
    ],
)
def test_downstream_commands_call_formal_runners(
    monkeypatch, tmp_path, command: str, runner_name: str, extra_args: list[str]
):
    captured: dict[str, object] = {}

    def fake_runner(output_dir, **kwargs):
        captured["output_dir"] = output_dir
        captured.update(kwargs)
        return {"status": "success"} if command == "run-xgb" else {}

    monkeypatch.setattr(cli, runner_name, fake_runner)
    monkeypatch.setattr(
        sys,
        "argv",
        _cli_args([command, "--output", str(tmp_path), *extra_args]),
    )

    cli.main()

    assert captured["output_dir"] == tmp_path
    if command == "run-causal-review":
        assert captured["control_columns"] == ["c1", "c2"]
        assert captured["maxlag"] == 5
        assert captured["min_rows"] == 60
        assert captured["top_n"] == 4
    if command == "run-xgb":
        assert captured["control_columns"] == ["c1"]
        assert captured["whitelist"] == ["w1", "w2"]
        assert captured["top_n"] == 6
        assert captured["max_lag"] == 12


@pytest.mark.parametrize(
    ("result", "expected", "unexpected"),
    [
        ({"status": "success"}, "XGB 四级验证完成。", "XGB 四级验证失败。"),
        (
            {"status": "invalid_input", "error_message": "bad input"},
            "XGB 四级验证失败。\nstatus=invalid_input\nerror=bad input",
            "XGB 四级验证完成。",
        ),
    ],
)
def test_run_xgb_cli_reports_runner_status(
    monkeypatch, tmp_path, capsys, result, expected, unexpected
):
    monkeypatch.setattr(cli, "run_xgb_for_active_branch", lambda *args, **kwargs: result)
    monkeypatch.setattr(
        sys, "argv", _cli_args(["run-xgb", "--output", str(tmp_path)])
    )

    cli.main()

    output = capsys.readouterr().out
    assert expected in output
    assert unexpected not in output


# --- 13. legacy enable flags on processed mode must not auto-select --------


def test_processed_analyze_with_enable_flags_rejects_without_confirm_branch(
    monkeypatch, tmp_path
):
    input_path = _write_input(tmp_path)
    monkeypatch.setattr(
        cli,
        "run_initial_screening_workflow",
        lambda config: {
            "raw": {},
            "processed": {},
            "comparison_path": Path(config.output_dir) / "preprocessing_comparison.csv",
            "context_path": Path(config.output_dir) / "preprocessing_context.json",
        },
    )
    monkeypatch.setattr(
        cli,
        "run_granger_for_active_branch",
        lambda output_dir: pytest.fail("must not run downstream before confirm-branch"),
    )
    monkeypatch.setattr(
        cli,
        "run_model_for_active_branch",
        lambda output_dir: pytest.fail("must not run downstream before confirm-branch"),
    )
    monkeypatch.setattr(sys, "argv", _cli_args([
        "analyze",
        "--input", str(input_path),
        "--time-column", "time",
        "--target", "target",
        "--output", str(tmp_path / "out"),
        "--preprocess-mode", "lowpass",
        "--enable-granger",
        "--enable-model",
    ]))

    with pytest.raises(SystemExit, match="请先 confirm-branch"):
        cli.main()


def test_raw_analyze_with_enable_flags_runs_formal_runners_after_promotion(
    monkeypatch, tmp_path
):
    input_path = _write_input(tmp_path)
    monkeypatch.setattr(
        cli,
        "run_initial_screening_workflow",
        lambda config: {"branch": "raw", "timings": {}},
    )
    calls: list[str] = []
    monkeypatch.setattr(
        cli,
        "run_granger_for_active_branch",
        lambda output_dir: calls.append(f"granger:{output_dir}"),
    )
    monkeypatch.setattr(
        cli,
        "run_model_for_active_branch",
        lambda output_dir: calls.append(f"model:{output_dir}"),
    )
    monkeypatch.setattr(sys, "argv", _cli_args([
        "analyze",
        "--input", str(input_path),
        "--time-column", "time",
        "--target", "target",
        "--output", str(tmp_path / "out"),
        "--preprocess-mode", "raw",
        "--enable-granger",
        "--enable-model",
    ]))

    cli.main()

    assert calls == [f"granger:{tmp_path / 'out'}", f"model:{tmp_path / 'out'}"]


# --- guard: tests never train XGB for real --------------------------------


def test_cli_test_suite_never_imports_legacy_xgb_runner_for_training():
    source = Path("chem_ts_corr/cli.py").read_text(encoding="utf-8")
    assert "run_xgb_analysis" not in source
    assert "run_xgb_for_active_branch(" in source
