from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import (
    KEY_FIELDS,
    REPORT_TOP_K,
    evaluate_case_expectations,
    evaluate_noise_false_positives,
    evaluate_rank_stability,
    metrics,
    run_case,
)
from .four_layer_cases import CASES, SyntheticCase


BASELINE_PATH = Path("tests/baselines/initial_screening_baseline.json")
STABILITY_SCENARIOS = {
    "true_lagged_driver",
    "noise_only",
    "mixed_evidence",
    "collinear_proxy",
}


def _json_value(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if pd.isna(value) else float(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]
    if value is None or pd.isna(value):
        return None
    return value


def build_case_report(
    case: SyntheticCase,
    ranked: pd.DataFrame,
    stability: dict[str, object] | None = None,
) -> dict[str, object]:
    evidence = [
        {field: _json_value(row.get(field)) for field in KEY_FIELDS}
        for row in ranked.to_dict("records")
    ]
    expectation = evaluate_case_expectations(case, ranked, stability)
    return {
        "scenario": case.metadata["scenario"],
        "seed": case.metadata["seed"],
        "samples": case.metadata["n"],
        "parameters": _json_value(case.metadata),
        "variable_types": {
            key: sorted(values) for key, values in case.variable_types.items()
        },
        "reference_map": case.reference_map,
        "top_k": REPORT_TOP_K,
        "top_k_results": ranked.head(REPORT_TOP_K)["variable"].tolist(),
        "key_evidence": evidence,
        "metrics": _json_value(metrics(case, ranked)),
        "stability_metrics": _json_value(stability or {}),
        **expectation,
    }


def build_full_baseline_report(output_root: Path) -> dict[str, object]:
    report: dict[str, object] = {}
    for name, factory in CASES.items():
        case = factory()
        ranked = run_case(case, output_root / name / "base")
        stability = (
            evaluate_rank_stability(factory, output_root / name / "stability")
            if name in STABILITY_SCENARIOS
            else None
        )
        if name == "noise_only":
            stability["multi_seed_false_positives"] = evaluate_noise_false_positives(
                factory,
                output_root / name / "multi_seed",
            )
        report[name] = build_case_report(case, ranked, stability)
    return report


def main() -> None:
    report = build_full_baseline_report(Path(".pytest_cache") / "synthetic_baseline")
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
