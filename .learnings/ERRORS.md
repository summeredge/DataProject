# Errors

## ERR-20260720-001 - 工况裁行发生在滞后构造之前

**Scope**
Project

**Area**
domain logic / tests

**Failure**
DatetimeIndex 保存了原始采样周期，但主筛查仍先删除工况外行，导致目标时刻所需的跨工况滞后来源不可用。

**Root Cause**
验证只覆盖了“不压缩缺口”，没有覆盖“在完整时间轴构造滞后、仅按当前目标时刻筛选工况”。

**Correction**
主筛查、二次验证和 XGB 验证保留完整时间轴，并把工况选择作为 `target_mask` 传入相关、模型、Granger 和 XGB 特征构造路径。

**Prevention Rule**
涉及工况筛选的滞后算法必须用交替工况回归测试，确认来源时刻可位于目标工况之外。

**Promotion Decision**
Do not promote

**Test Decision**
Regression test added

**Related Files**
- chem_ts_corr/service.py
- chem_ts_corr/preprocess.py
- chem_ts_corr/web.py
- chem_ts_corr/xgb_runner.py
- chem_ts_corr/xgb_validation.py
- tests/test_service_metrics.py
- tests/test_time_axis_lag_semantics.py
- tests/test_xgb_feature_contract.py
- tests/test_xgb_web.py
