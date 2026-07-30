from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs

import pandas as pd
import pytest

from chem_ts_corr import web
from chem_ts_corr.web import INDEX_HTML


def test_web_data_input_uses_automatic_encoding_without_a_manual_selector():
    file_input = INDEX_HTML.index('id="fileInput"')
    upload_button = INDEX_HTML.index('id="upload"')

    assert file_input < upload_button
    assert 'id="encoding"' not in INDEX_HTML
    assert 'encoding=auto' in INDEX_HTML
    assert 'form.append("encoding", "auto")' in INDEX_HTML


def _write_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, values: list[object]
) -> str:
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    file_id = "0123456789abcdef0123456789abcdef"
    pd.DataFrame({"Unnamed: 0": values, "target": range(len(values))}).to_csv(
        uploads / f"{file_id}.csv", index=False, encoding="utf-8-sig"
    )
    monkeypatch.setattr(web, "UPLOADS_DIR", uploads)
    return file_id


def _trend_params(file_id: str, time_column: str) -> dict[str, list[str]]:
    return parse_qs(
        "&".join(
            [
                f"file_id={file_id}",
                "encoding=utf-8-sig",
                f"time_column={time_column}",
                "variables=target",
                "trend_max_points=10000",
                "segment_mode=all",
                "preprocess_mode=raw",
                "detrend_window=24",
                "excluded_columns=",
            ]
        ),
        keep_blank_values=True,
    )


def test_unnamed_first_column_is_auto_detected_as_time_column(tmp_path, monkeypatch):
    file_id = _write_upload(
        tmp_path,
        monkeypatch,
        ["2026-01-01 00:00", "2026-01-01 00:01"],
    )

    response = web._columns_response(file_id, "utf-8-sig")

    assert response["timeColumn"] == "Unnamed: 0"
    assert response["autoTimeColumn"] == "Unnamed: 0"


def test_numeric_unnamed_first_column_is_not_auto_detected(tmp_path, monkeypatch):
    file_id = _write_upload(tmp_path, monkeypatch, [0, 1, 2, 3])

    response = web._columns_response(file_id, "utf-8-sig")

    assert "timeColumn" not in response


def test_explicit_time_column_keeps_existing_name_based_detection(tmp_path):
    path = tmp_path / "input.csv"
    pd.DataFrame({"time": ["not-a-time", "also-not-a-time"], "target": [1, 2]}).to_csv(
        path, index=False, encoding="utf-8-sig"
    )
    sample, encoding = web._read_data_sample(path, "utf-8-sig")

    response = web._time_range_metadata(path, sample, encoding)

    assert response == {"timeColumn": "time"}


def test_auto_detected_time_column_drives_trend_x_axis(tmp_path, monkeypatch):
    values = [f"2026-01-01 00:{minute:02d}" for minute in range(10)]
    file_id = _write_upload(tmp_path, monkeypatch, values)
    detected_column = web._columns_response(file_id, "utf-8-sig")["timeColumn"]

    response = web._trend_response(_trend_params(file_id, detected_column))

    assert [point["x"] for point in response["series"][0]["points"]] == [
        f"2026-01-01 00:{minute:02d}:00" for minute in range(10)
    ]


@pytest.mark.parametrize("name", ["", "Unnamed: 0", "index", "index_0"])
def test_implicit_time_column_names_are_detected_only_in_the_first_column(
    monkeypatch, name
):
    sample = pd.DataFrame({name: ["2026-01-01", "2026-01-02"], "target": [1, 2]})

    monkeypatch.setattr(
        web,
        "read_timeseries_table",
        lambda *args, **kwargs: (sample.loc[:, kwargs["usecols"]], "utf-8-sig"),
    )

    response = web._time_range_metadata(Path("unused.csv"), sample, "utf-8-sig")

    assert response["timeColumn"] == name


def test_implicit_time_name_in_a_later_column_is_not_considered(monkeypatch):
    sample = pd.DataFrame(
        {
            "target": [1, 2],
            "Unnamed: 0": ["2026-01-01", "2026-01-02"],
        }
    )
    monkeypatch.setattr(
        web,
        "read_timeseries_table",
        lambda *args, **kwargs: (sample.loc[:, kwargs["usecols"]], "utf-8-sig"),
    )

    assert web._time_range_metadata(Path("unused.csv"), sample, "utf-8-sig") == {}


def test_auto_detection_requires_mostly_parseable_increasing_values():
    parsed = pd.Series(pd.date_range("2026-01-01", periods=21, freq="min"))
    numeric_values = pd.Series([0, 1, 2, 3])
    parseable_at_threshold = parsed.astype(str).iloc[:-1].tolist() + ["not-a-time"]
    below_parse_threshold = parsed.astype(str).iloc[:-2].tolist() + ["not-a-time"] * 2
    one_out_of_order = parsed.copy()
    one_out_of_order.iloc[10:12] = one_out_of_order.iloc[[11, 10]].to_numpy()
    two_out_of_order = parsed.copy()
    two_out_of_order.iloc[10:13] = two_out_of_order.iloc[[12, 11, 10]].to_numpy()

    assert web._is_high_quality_time_series(parsed, parsed.astype(str))
    assert not web._is_high_quality_time_series(
        pd.to_datetime(numeric_values), numeric_values
    )
    assert web._is_high_quality_time_series(
        pd.Series(pd.to_datetime(parseable_at_threshold, errors="coerce")),
        pd.Series(parseable_at_threshold),
    )
    assert not web._is_high_quality_time_series(
        pd.Series(pd.to_datetime(below_parse_threshold, errors="coerce")),
        pd.Series(below_parse_threshold),
    )
    assert web._is_high_quality_time_series(
        one_out_of_order, one_out_of_order.astype(str)
    )
    assert not web._is_high_quality_time_series(
        two_out_of_order, two_out_of_order.astype(str)
    )


def test_frontend_marks_implicit_time_column_as_auto_detected():
    load_columns = INDEX_HTML.split("async function loadColumns()", 1)[1].split(
        "async function analyze()", 1
    )[0]

    assert "data.autoTimeColumn" in load_columns
    assert 'el("timeColumn").value = data.timeColumn' in load_columns
