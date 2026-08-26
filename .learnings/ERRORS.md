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

## ERR-20260731-001 - 单选变量筛选框不应常驻显示

**Scope**
Project

**Area**
UI / workflow

**Failure**
将单选变量的筛选输入框直接放在页面表单中，虽然能够过滤变量，但增加了常态页面高度并偏离用户期望的下拉交互。

**Root Cause**
只按“标题、筛选框、下拉框”的静态结构实现，没有确认筛选框应位于点击后展开的下拉层中。

**Correction**
单选变量组件关闭时仅显示当前值，点击后才显示筛选框和候选列表；原生 `select` 继续保存和提交变量值。

**Prevention Rule**
为现有下拉框增加搜索时，默认把搜索框放入折叠下拉层；除非需求明确要求常驻搜索框，不得增加表单常态高度。

**Promotion Decision**
Do not promote

**Test Decision**
Regression test added

**Related Files**
- chem_ts_corr/web.py
- tests/test_web_variable_select_search.py

## [ERR-20260814-001] pytest 目标文件不存在

**Priority**: low
**Status**: resolved
**Area**: tools

### 摘要
定向回归命令引用了不存在的 `tests/test_preprocess.py`，pytest 在收集前退出。

### 错误信息
```
ERROR: file or directory not found: tests/test_preprocess.py
```

### 上下文
- 执行预处理与排除窗口定向测试时手工指定了错误测试文件名。

### 建议修复
在指定测试文件前，先用项目文件检索确认实际测试文件名；本次已改用现有的
`test_lowpass_preprocessing.py`、`test_diff_interval_preprocessing.py` 与
`test_causal_preprocessing.py` 并通过。

### 元数据
- Reproducible: yes

---

## [ERR-20260826-001] FastCtx MCP 传输通道关闭

**Priority**: low
**Status**: resolved
**Area**: tools

### 摘要
本次会话直接调用 FastCtx 的 `inspect_local_file` 与 `glob` 时均因传输通道关闭而失败。

### 错误信息
```
tool call failed for `fastctx/inspect_local_file`
Caused by:
    Transport closed
```

### 上下文
- 读取任务附件和仓库文件时，FastCtx 独立 MCP 调用连续返回 `Transport closed`。
- 后续改用只读 PowerShell 读取，未影响实现和验证。

### 建议修复
FastCtx 传输失败后使用等价的只读本地文件读取路径；不要因此修改项目代码或测试语义。

### 元数据
- Reproducible: unknown
- See Also: ERR-20260818-001

---

## [ERR-20260826-002] 子智能体调用参数组合不兼容

**Priority**: low
**Status**: resolved
**Area**: tools

### 摘要
调用指定 `luna_worker` 时同时设置完整历史 fork，工具拒绝了该参数组合。

### 错误信息
```
Full-history forked agents inherit the parent agent type; omit agent_type, or spawn without a full-history fork.
```

### 上下文
- 初次尝试以 `agent_type=luna_worker` 和 `fork_context=true` 启动实现任务。
- 随后改为不 fork 历史的指定 worker 调用；用户要求后续不再调用任何子智能体。

### 建议修复
指定 agent type 时不要同时启用 full-history fork；如果用户要求停止子智能体，立即停止后续调用并由主线程继续。

### 元数据
- Reproducible: yes

---

## [ERR-20260818-001] FastCtx 不能作为 functions.exec 的嵌套工具调用

**Priority**: low
**Status**: resolved
**Area**: tools

### 摘要
尝试在 `functions.exec` 中并行调用 `tools.mcp__fastctx__inspect_local_file` 时，运行时报告该方法不存在。

### 错误信息
```
TypeError: tools.mcp__fastctx__inspect_local_file is not a function
```

### 上下文
- FastCtx 在当前会话中作为独立 MCP 命名空间提供，而不是 `functions.exec` 的嵌套工具。

### 建议修复
直接调用 `mcp__fastctx.inspect_local_file` / `grep` / `glob`；仅把工具声明明确列入
`functions.exec` 的方法放入聚合脚本。

### 元数据
- Reproducible: yes

---
