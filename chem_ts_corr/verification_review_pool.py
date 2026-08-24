from __future__ import annotations

from pathlib import Path

import pandas as pd


FILENAME = "verification_review_pool.csv"
COLUMNS = ["variable", "candidate_source", "source_rank", "include_reason"]
SOURCES = frozenset({"initial_screening", "manual_include", "model_discovery"})


def build_initial_verification_review_pool(
    ranked_features: pd.DataFrame,
    *,
    top_k: int,
    manual_include: list[str] | None = None,
) -> pd.DataFrame:
    """Create a separate second-stage review pool without changing screening outputs."""
    ranked = _ranked_rows(ranked_features)
    initial = ranked.head(max(0, int(top_k)))
    rows = [
        {
            "variable": row["variable"],
            "candidate_source": "initial_screening",
            "source_rank": row["source_rank"],
            "include_reason": "Top-K进入",
        }
        for row in initial.to_dict(orient="records")
    ]
    known_ranks = dict(zip(ranked["variable"], ranked["source_rank"]))
    existing = {row["variable"] for row in rows}
    for variable in dict.fromkeys(str(value) for value in (manual_include or []) if value):
        if variable in existing:
            continue
        rows.append(
            {
                "variable": variable,
                "candidate_source": "manual_include",
                "source_rank": known_ranks.get(variable, pd.NA),
                "include_reason": "初始人工指定",
            }
        )
        existing.add(variable)
    return _frame(rows)


def write_initial_verification_review_pool(
    run_dir: Path,
    ranked_features: pd.DataFrame,
    *,
    top_k: int,
    manual_include: list[str] | None = None,
) -> pd.DataFrame:
    pool = build_initial_verification_review_pool(
        ranked_features,
        top_k=top_k,
        manual_include=manual_include,
    )
    _write(run_dir, pool)
    return pool


def read_verification_review_pool(run_dir: Path) -> pd.DataFrame | None:
    path = Path(run_dir) / FILENAME
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError, pd.errors.ParserError):
        return _frame([])
    return _frame(frame.to_dict(orient="records"))


def add_to_verification_review_pool(
    run_dir: Path,
    ranked_features: pd.DataFrame,
    *,
    variable: str,
    candidate_source: str,
) -> pd.DataFrame:
    if candidate_source not in {"manual_include", "model_discovery"}:
        raise ValueError("verification_candidate_source_invalid")
    normalized = str(variable).strip()
    if not normalized:
        raise ValueError("verification_candidate_variable_required")
    ranked = _ranked_rows(ranked_features)
    known_ranks = dict(zip(ranked["variable"], ranked["source_rank"]))
    if normalized not in known_ranks:
        raise ValueError("verification_candidate_variable_not_in_initial_screening")
    pool = read_verification_review_pool(run_dir)
    if pool is None:
        pool = _frame([])
    if normalized in set(pool["variable"]):
        return pool
    row = {
        "variable": normalized,
        "candidate_source": candidate_source,
        "source_rank": known_ranks[normalized],
        "include_reason": (
            "人工加入复核池"
            if candidate_source == "manual_include"
            else "模型发现后人工确认"
        ),
    }
    pool = _frame([*pool.to_dict(orient="records"), row])
    _write(run_dir, pool)
    return pool


def pool_variables(pool: pd.DataFrame | None) -> list[str] | None:
    if pool is None:
        return None
    return [str(value) for value in pool["variable"].tolist() if str(value)]


def _ranked_rows(ranked_features: pd.DataFrame) -> pd.DataFrame:
    if ranked_features.empty or "variable" not in ranked_features.columns:
        return pd.DataFrame(columns=["variable", "source_rank"])
    ranked = ranked_features[["variable"]].copy(deep=True)
    ranks = pd.to_numeric(ranked_features.get("driver_rank"), errors="coerce")
    if ranks is None or ranks.isna().all():
        ranks = pd.Series(range(1, len(ranked) + 1), index=ranked.index, dtype="Int64")
    ranked["source_rank"] = ranks
    ranked = ranked.dropna(subset=["variable", "source_rank"])
    ranked["variable"] = ranked["variable"].astype(str)
    ranked = ranked[ranked["variable"].str.strip().ne("")]
    return ranked.sort_values("source_rank", kind="mergesort").drop_duplicates(
        subset=["variable"], keep="first"
    )


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=COLUMNS)
    if frame.empty:
        return pd.DataFrame(columns=COLUMNS)
    frame["variable"] = frame["variable"].astype(str)
    frame["candidate_source"] = frame["candidate_source"].astype(str)
    frame["source_rank"] = pd.to_numeric(frame["source_rank"], errors="coerce").astype("Int64")
    frame["include_reason"] = frame["include_reason"].fillna("").astype(str)
    return frame[COLUMNS]


def _write(run_dir: Path, pool: pd.DataFrame) -> None:
    path = Path(run_dir) / FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    _frame(pool.to_dict(orient="records")).to_csv(path, index=False, encoding="utf-8-sig")
