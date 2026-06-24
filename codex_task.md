# Codex Task: fix causal review P1 issues

Base branch: `codex/v0.5-causal-review`

Please fix only the following P1 issues:

1. Limit causal review evidence and final summary to the selected candidates after `top_n` and risk-flag filtering.
2. Do not convert negative screening lags to positive lags when building conditional Granger ranked windows.
3. Exclude the current candidate variable from `control_columns` inside conditional Granger baseline construction.

Constraints:

- Keep Windows compatibility.
- Do not change existing CSV schemas unless needed for these P1 fixes.
- Do not change Chinese UI labels or table mappings.
- Do not reword predictive validation as deterministic causality.
- Do not refactor unrelated code.
- Add or update focused tests for the three issues above.
