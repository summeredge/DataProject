# Errors

## ERR-20260727-002 - 受限会话拒绝启动 Python 解释器

**Scope**
Project

**Area**
tests / environment

**Failure**
受限 Codex PowerShell 会话执行已验证的 Python 3.11 路径时返回 `拒绝访问`，pytest 尚未启动。

**Root Cause**
Windows 进程启动权限由受限会话策略阻止；不能据此判断 Python 未安装或测试代码失败。

**Correction**
保留已验证解释器路径，并改为请求真实 Windows 执行路径运行测试。

**Prevention Rule**
遇到解释器启动拒绝时区分环境权限与测试结果，使用已验证解释器请求受限会话外执行，不改用 `py` 的失败结果作结论。

**Promotion Decision**
Do not promote

**Test Decision**
Regression test recommended

**Related Files**
- tests/test_initial_screening_contract.py

## ERR-20260727-001 - PowerShell 不支持 Unix 文件通配路径

**Scope**
Project

**Area**
tools / workflow

**Failure**
在 PowerShell 中将 `tests/test_*` 作为 `rg` 的字面路径参数，导致文件路径解析失败。

**Root Cause**
把 Unix shell 的路径通配写法直接用于 Windows PowerShell；`rg` 的文件筛选应直接使用目录参数和模式参数。

**Correction**
改用 `rg -n "pattern" tests`，不把 `tests/test_*` 作为路径传入。

**Prevention Rule**
PowerShell 下搜索仓库文件优先使用 `rg --files` 或将目录与 `-g` 模式分开传给 `rg`。

**Promotion Decision**
Do not promote

**Test Decision**
Not testable

**Related Files**
- tests

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
## ERR-20260727-003: 组合回归测试超时

- **Date**: 2026-07-27
- **Context**: 在已授权的 Windows Python 解释器下，将多个初筛、服务和合成基线测试放在同一次 pytest 调用中。
- **Failure**: 命令超过超时时间，未返回测试汇总。
- **Lesson**: 对涉及模型或合成场景的回归测试拆分执行，先用小批次确认契约，再单独运行高成本场景。
## ERR-20260727-004: 合成基线全文件回归超时

- **Date**: 2026-07-27
- **Context**: 在包含多轮稳定性计算和 committed baseline 重算的合成测试文件上运行完整 pytest。
- **Failure**: 命令超过超时时间，未返回测试汇总；拆分的非基线场景测试已通过。
- **Lesson**: 基线一致性测试与场景回归应分开运行，并为基线重算预留更长的超时窗口。
## ERR-20260727-005: 全套件暴露旧初筛契约断言

- **Date**: 2026-07-27
- **Context**: 完整 pytest 在初筛输出字段和排序边界调整后运行。
- **Failure**: 1179 个测试通过，24 个旧测试仍要求残差/稳定性/模型字段或 driver-rank/四层 UI 文案。
- **Lesson**: 初筛与后续阶段边界变化时，需同步更新契约测试；后续模块的独立计算测试应保留在对应接口或结果文件层验证。
## ERR-20260727-006: 受影响测试组合命令输出通道异常

- **Date**: 2026-07-27
- **Context**: 一次性运行多个受影响的评分、稳定性和 UI 契约测试文件。
- **Failure**: 执行器超时并在 pytest 刷新 stdout 时报告 `OSError: [Errno 22] Invalid argument`，未返回测试汇总。
- **Lesson**: 将回归按模块拆成更小命令，避免长输出或执行器通道异常掩盖实际测试结果。
## ERR-20260727-007: 全套件剩余一条稳定性字段旧断言

- **Date**: 2026-07-27
- **Context**: 修正前的完整 pytest 回归。
- **Failure**: 1202 个测试通过，仅 `test_screening_score_v2.py` 仍直接读取已从初筛输出移除的 `stability_score`。
- **Lesson**: 稳定性保留在独立分析结果和内部评分计算中，初筛输出契约应只验证不暴露该字段。
