from __future__ import annotations

import json
from pathlib import Path

from .evaluate import KEY_FIELDS, metrics, run_case
from .four_layer_cases import CASES


BASELINE_PATH = Path("tests/baselines/four_layer_ranking_baseline.json")
FAILURES = {
    "downstream_response": "layer_2 target_leads_variable does not cap candidate_grade",
    "common_driver": "layer_3 common_capacity_driver is not emitted for x_common",
    "lag_boundary_artifact": "layer_2 lag_boundary does not cap candidate_grade",
    "mixed_evidence": "layer_3 common driver suppression and layer_2 downstream grade cap are absent",
    "nonlinear_stable_driver": "layer_1 Pearson/Spearman lag screening cannot detect U-shaped nonlinear evidence",
}


def build_case_report(case, ranked):
    evidence = ranked.reindex(columns=KEY_FIELDS).where(lambda frame: frame.notna(), None).to_dict("records")
    return {"scenario": case.metadata["scenario"], "seed": case.metadata["seed"], "samples": case.metadata["n"], "parameters": case.metadata, "top_k_results": ranked["variable"].tolist(), "key_evidence": evidence, "metrics": metrics(case, ranked), "passed": case.metadata["scenario"] not in FAILURES, "failure_reason": FAILURES.get(case.metadata["scenario"], "")}


def build_full_baseline_report(output_root: Path) -> dict:
    return {name: build_case_report(case := factory(), run_case(case, output_root / name)) for name, factory in CASES.items()}


def main() -> None:
    report = build_full_baseline_report(Path(".pytest_cache") / "synthetic_baseline")
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
