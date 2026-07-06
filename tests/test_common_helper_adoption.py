from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_benjamini_hochberg_is_imported_not_redefined():
    for path in ["chem_ts_corr/lag.py", "chem_ts_corr/causality.py", "chem_ts_corr/conditional_granger.py"]:
        source = _source(path)
        assert "from chem_ts_corr.common import benjamini_hochberg" in source
        assert "def _benjamini_hochberg" not in source


def test_scalar_conversion_no_longer_uses_single_value_series():
    for path in ROOT.glob("chem_ts_corr/*.py"):
        assert "pd.Series([value])" not in path.read_text(encoding="utf-8")


def test_left_join_missing_is_imported_not_redefined():
    for path in ["chem_ts_corr/causal_review_evidence.py", "chem_ts_corr/causal_review_service.py"]:
        source = _source(path)
        assert "left_join_missing" in source
        assert "from chem_ts_corr.common import" in source
        assert "def _left_join_missing" not in source
        assert "def left_join_missing" not in source
