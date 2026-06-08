from pathlib import Path

from chem_ts_corr.data import load_timeseries_csv, read_timeseries_table


def test_read_timeseries_table_accepts_txt_with_inferred_separator(tmp_path: Path):
    path = tmp_path / "sample.txt"
    path.write_text(
        "time target x\n"
        "2024-01-01 1 2\n"
        "2024-01-02 2 3\n"
        "2024-01-03 3 4\n"
        "2024-01-04 4 5\n"
        "2024-01-05 5 6\n"
        "2024-01-06 6 7\n"
        "2024-01-07 7 8\n"
        "2024-01-08 8 9\n"
        "2024-01-09 9 10\n"
        "2024-01-10 10 11\n",
        encoding="utf-8",
    )

    table, used_encoding = read_timeseries_table(path, encoding="auto")
    frame = load_timeseries_csv(path, time_column="time", encoding="auto")

    assert used_encoding == "utf-8-sig"
    assert table.columns.tolist() == ["time", "target", "x"]
    assert frame.index.name == "time"
    assert frame.columns.tolist() == ["target", "x"]
