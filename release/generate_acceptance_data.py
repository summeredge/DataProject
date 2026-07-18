"""Generate deterministic, non-production data for Windows acceptance tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260718


def _small_frame() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = 240
    timestamps = pd.date_range("2026-01-01", periods=rows, freq="5min").to_numpy()
    timestamps[120] = timestamps[119]
    driver_1 = np.cumsum(rng.normal(0, 0.8, rows))
    driver_2 = rng.normal(0, 1, rows)
    noise = rng.normal(0, 1, rows)
    target = 0.82 * np.roll(driver_1, 3) + 0.18 * driver_2 + rng.normal(0, 0.25, rows)
    target[:3] = target[3]
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "target": target,
            "driver_1": driver_1,
            "driver_2": driver_2,
            "noise": noise,
        }
    )
    frame.loc[[17, 91], "driver_2"] = np.nan
    frame.loc[[45, 173], "noise"] = np.nan
    return frame


def _chinese_frame(small: pd.DataFrame) -> pd.DataFrame:
    return small.rename(
        columns={
            "timestamp": "时间",
            "target": "目标变量",
            "driver_1": "温度",
            "driver_2": "压力",
            "noise": "流量",
        }
    )


def _large_frame(rows: int, variables: int) -> pd.DataFrame:
    if rows < 200:
        raise ValueError("large data must contain at least 200 rows")
    if variables < 4:
        raise ValueError("large data must contain at least 4 numeric variables")

    rng = np.random.default_rng(SEED)
    base = np.cumsum(rng.normal(0, 0.15, rows))
    numeric: dict[str, np.ndarray] = {}
    target = 0.75 * np.roll(base, 8) + rng.normal(0, 0.35, rows)
    target[:8] = target[8]
    numeric["target"] = target
    for index in range(1, variables):
        if index <= 6:
            lag = index * 2
            values = (0.8 - index * 0.05) * np.roll(base, lag) + rng.normal(0, 0.5, rows)
            values[:lag] = values[lag]
            numeric[f"signal_{index:02d}"] = values
        else:
            numeric[f"noise_{index:02d}"] = rng.normal(0, 1, rows)
    return pd.DataFrame(
        {"timestamp": pd.date_range("2026-01-01", periods=rows, freq="min"), **numeric}
    )


def generate(output_dir: Path, large_rows: int, large_variables: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    small = _small_frame()
    chinese = _chinese_frame(small)
    large = _large_frame(large_rows, large_variables)

    small.to_csv(output_dir / "acceptance_small.csv", index=False, encoding="utf-8-sig")
    small.to_csv(output_dir / "acceptance_small.txt", index=False, sep="\t", encoding="utf-8-sig")
    small.to_csv(output_dir / "acceptance_small.tsv", index=False, sep="\t", encoding="utf-8-sig")
    small.to_excel(output_dir / "acceptance_small.xlsx", index=False, engine="openpyxl")
    chinese.to_csv(
        output_dir / "acceptance_chinese_columns.csv", index=False, encoding="utf-8-sig"
    )
    chinese.to_excel(
        output_dir / "acceptance_chinese_columns.xlsx", index=False, engine="openpyxl"
    )
    large.to_csv(output_dir / "acceptance_large.csv", index=False, encoding="utf-8-sig")

    print(f"Generated acceptance data in: {output_dir.resolve()}")
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            print(f"{path.name}: {path.stat().st_size} bytes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "test-data")
    parser.add_argument("--large-rows", type=int, default=45000)
    parser.add_argument("--large-variables", type=int, default=40)
    args = parser.parse_args()
    generate(args.output_dir, args.large_rows, args.large_variables)


if __name__ == "__main__":
    main()
