import inspect
import os
import time

import pandas as pd

from chem_ts_corr import web


def _make_config(tmp_path, roles_path=None):
    data_path = tmp_path / "sample.csv"
    pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=20, freq="min"),
            "target": range(20),
            "x": range(20, 40),
        }
    ).to_csv(data_path, index=False, encoding="utf-8-sig")
    return web.AnalysisConfig(
        input_path=data_path,
        time_column="time",
        target="target",
        output_dir=tmp_path / "out",
        min_valid_ratio=0.0,
        roles_path=roles_path,
    )


def test_scaled_frame_cache_key_includes_roles_path_and_file_state(tmp_path):
    roles_a = tmp_path / "roles_a.csv"
    roles_b = tmp_path / "roles_b.csv"
    roles_a.write_text("variable,role\nx,PV\n", encoding="utf-8")
    roles_b.write_text("variable,role\nx,IGNORE\n", encoding="utf-8")

    key_without_roles = web._scaled_frame_cache_key(_make_config(tmp_path, None))
    key_a = web._scaled_frame_cache_key(_make_config(tmp_path, roles_a))
    key_b = web._scaled_frame_cache_key(_make_config(tmp_path, roles_b))

    assert key_without_roles != key_a
    assert key_a != key_b
    assert str(roles_a.resolve()) in key_a
    assert str(roles_b.resolve()) in key_b

    old_key = key_a
    time.sleep(0.01)
    roles_a.write_text("variable,role\nx,IGNORE\n", encoding="utf-8")
    os.utime(roles_a, None)
    new_key = web._scaled_frame_cache_key(_make_config(tmp_path, roles_a))
    assert old_key != new_key


def test_scaled_frame_cache_key_source_mentions_roles_path_and_stat():
    source = inspect.getsource(web._scaled_frame_cache_key)
    assert "roles_path" in source
    assert "roles_stat" in source or "role_stat" in source
    assert "st_mtime_ns" in source
    assert "st_size" in source


def test_secondary_scaled_frame_cache_invalidates_when_roles_file_changes(monkeypatch, tmp_path):
    web._clear_scaled_frame_cache()
    roles = tmp_path / "roles.csv"
    roles.write_text("variable,role\nx,PV\n", encoding="utf-8")
    config = _make_config(tmp_path, roles)

    calls = {"n": 0}
    original_loader = web.load_timeseries_csv

    def counting_loader(*args, **kwargs):
        calls["n"] += 1
        return original_loader(*args, **kwargs)

    monkeypatch.setattr(web, "load_timeseries_csv", counting_loader)
    first = web._scaled_frame_for_secondary(config)
    second = web._scaled_frame_for_secondary(config)
    assert calls["n"] == 1
    assert first.equals(second)

    time.sleep(0.01)
    roles.write_text("variable,role\nx,IGNORE\n", encoding="utf-8")
    os.utime(roles, None)
    third = web._scaled_frame_for_secondary(config)
    assert calls["n"] == 2
    assert "x" not in third.columns
