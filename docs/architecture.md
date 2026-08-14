# Architecture

## 分析流程

数据输入 → 数据预处理 → 第一阶段：主筛查 → 候选池 → 第二阶段：增强验证 →
第三阶段：综合复核 → 第四阶段：时间外预测验证

## 阶段边界

主筛查负责生成候选。

后续分析只能提供补充证据，不得修改：

-   final_score
-   初筛排序
-   Top-K

## 数据流

允许：

主筛查 → 候选池 → 后续验证

禁止：

-   模型结果 → 初筛评分
-   综合复核 → 覆盖初筛结果
-   未执行分析 → 展示分析结论

## 排除窗口（PR-TR1 / PR-TR2 / PR-TR3）

完整上传数据始终保留。`exclude_windows` 只是本次分析的数据选择条件：在重采样和
预处理之前按闭区间 `start <= timestamp <= end` 过滤；清空 `exclude_windows` 即等价于
恢复全部数据。排除窗口不是风险标签，不得改变 `final_score` 或评分算法。

趋势页维护当前上传数据上下文的 `exclude_windows`，按上传文件与时间列隔离；新上传数据
不会继承旧窗口。加入、恢复或恢复全部只改变该待分析状态，趋势仍展示完整上传数据。

Web 启动分析时将窗口快照写入本次 `AnalysisConfig` 和 `run_config.json`。正式数据流固定为：

```text
完整上传数据 → apply_exclude_windows → 重采样 → 预处理 → 初筛 → 后续阶段
```

Raw 与 Processed 分支通过同一数据入口获取同一批排除后的时间点；排除形成的物理断点不得
被重采样插值、前向填充、低通状态、去趋势或滞后/差分跨越。`preprocessing_context.json`
冻结窗口及其统计；所有 downstream runner 仅从该 context 构造配置，之后在 Web 中改变窗口
不会改变已创建运行或其后续阶段输入。

## 预处理模式

后端 `transform_frame()` / `transform_frame_causal()` 支持：

```text
raw
lowpass
lowpass_detrend
lowpass_diff
```

旧模式（`detrend` / `diff` / `detrend_diff`）继续保留 backend compatibility，
但不再出现在正式 Web / CLI 初筛选择中，也不得静默映射为新模式。
模式语义与配置字段约束见 `docs/contracts.md`。

当前正式 Web / CLI 预处理模式：

```text
raw
lowpass
lowpass_detrend
lowpass_diff
```

Web 与 CLI 的 `analyze` 入口统一走
`run_initial_screening_workflow()`：`raw` 只运行 Raw 并自动发布正式初筛；
任一 `lowpass*` 模式同时运行 Raw + 该模式两个独立初筛，生成
`preprocessing_comparison.csv` 并进入 `awaiting_confirmation`，等待用户通过
`confirm_initial_screening_branch()` 明确确认正式分支。

单 branch 初筛 runner 已实现：

```text
run_initial_screening_branch():
  branch=raw       → preprocess_mode 必须为 raw
  branch=processed → preprocess_mode 只能为 lowpass / lowpass_detrend / lowpass_diff
```

一次调用只执行一个分支，结果写入：

```text
run_directory/
└─ screening_branches/
   ├─ raw/
   └─ processed/
```

非 Raw 双分支对比入口已实现：

```text
run_initial_screening_comparison():
  preprocess_mode 只能为 lowpass / lowpass_detrend / lowpass_diff
  → raw branch + selected processed branch 独立执行
  → 双分支均成功后生成 run_directory/preprocessing_comparison.csv
```

双分支执行不决定采用哪个分支。

统一 workflow 已实现：

```text
run_initial_screening_workflow():
  preprocess_mode 只能为 raw / lowpass / lowpass_detrend / lowpass_diff
  旧模式（detrend / diff / detrend_diff）必须明确拒绝
  完整上传数据 → apply_exclude_windows → 重采样 → 预处理
  → raw:
      仅运行 raw branch
      → 事务性 promotion 到正式 root
      → preprocessing_context.json 状态 not_required
  → lowpass*:
      复用 run_initial_screening_comparison()
      → raw + selected processed 双分支 + preprocessing_comparison.csv
      → preprocessing_context.json 状态 awaiting_confirmation
      → 正式 root 初筛文件不得存在
```

## 初筛双分支

第一阶段分支产物目标目录：

```text
run_directory/
├─ screening_branches/
│  ├─ raw/
│  └─ processed/
├─ preprocessing_comparison.csv
└─ preprocessing_context.json
```

隔离约束：

- 分支候选池不得合并；
- 不得按变量取两分支较高分数；
- 不得生成新的综合评分；
- 未确认前不得在运行根目录发布正式初筛文件；
- 确认后只发布选定分支的结果。

当前已实现：

- 单 branch 独立运行（`run_initial_screening_branch()`）；
- 非 Raw 双 branch orchestration（`run_initial_screening_comparison()`）；
- `preprocessing_comparison.csv` 对比产物；
- `run_initial_screening_workflow()` 统一 workflow 与
  `preprocessing_context.json`（raw `not_required` / 非 Raw
  `awaiting_confirmation` / `confirmed`）；
- 人工 branch confirmation（`confirm_initial_screening_branch()`）：只读取并
  验证已有 branch 文件后 promotion，确认 ≠ 重新运行初筛；
- 事务性 promotion：staging → backup → replace → context 更新，失败回滚，
  不产生 Raw/Processed 混合 root；
- downstream gate / lock：`begin_downstream_stage()` 读取 context，
  `awaiting_confirmation` 明确拒绝，`confirmed` / `not_required` 允许，首次
  通过后创建 `screening_downstream.lock`，lock 后禁止切换 branch；
- 增强筛选正式 branch/context 接入
  （`run_enhanced_screening_for_active_branch()`）：只消费正式 root 初筛结果
  与 `preprocessing_context.json` 的 active 预处理配置，`awaiting_confirmation`
  / 缺 context / 缺正式 root 输入时明确失败且不生成阶段结果，首次成功进入
  downstream 时创建 lock，lock 后其他 downstream stage 仍可继续；
- 普通 Granger 正式 branch/context 接入
  （`run_granger_for_active_branch()`）：与增强筛选相同的 context gate /
  lock 语义，只运行 ordinary/bivariate Granger 并沿用现有
  `granger_tests.csv` 输出；
- RF / SHAP / model discovery 正式 branch/context 接入
  （`run_model_for_active_branch()`）：与增强筛选相同的 context gate /
  lock 语义，只消费正式 root 的 `ranked_features.csv` /
  `risk_flags.csv` 等输入，生成 `shap_or_importance.csv`、
  `model_variable_importance.csv`、`model_discovered_candidates.csv`
  三个模型解释输出，不自动运行其他 downstream stage；
- conditional Granger / causal review / final review 正式 branch/context 接入
  （`run_causal_review_for_active_branch()`）：与增强筛选相同的 context
  gate / lock 语义，只消费正式 root 的 `ranked_features.csv`、
  `recommended_candidates.csv`、`causal_review_candidates.csv`、
  `risk_flags.csv`，复用现有 `run_causal_review_stage()`，生成
  `conditional_granger_scores.csv`、`causal_review_report.csv`、
  `causal_review_evidence.csv`、`final_review_summary.csv` 四个三级复核
  输出，不自动运行其他 downstream stage；
- 初筛允许为历史筛查使用回顾性预处理；增强筛选、普通 Granger、RF/SHAP
  与 conditional Granger/三级复核统一使用 causal preprocessing，禁止读取
  未来过程值；XGBoost 保持独立的 fold-safe causal preprocessing；
- `ranked_features.lag` 继续只表示历史初筛证据；正式 Enhanced、RF/SHAP
  与 conditional Granger 在 causal frame 上单独重算其 lag evidence，且只在
  内存中使用，不回写初筛文件。普通 Granger 已在 causal frame 全扫描 lag；
- branch 输出隔离到 `screening_branches/raw/` 或 `screening_branches/processed/`；
- branch runner 不向运行根目录发布正式初筛文件；
- 未锁定目录重新用于新 workflow 时先校验 mode，再清理旧 root 正式文件与旧
  context，避免暴露上一轮正式结果。

当前已实现（PR-12）：

- XGBoost 正式 branch/context 接入：`run_xgb_for_active_branch()` 复用
  `prepare_downstream_analysis_context()`，只消费正式 root 的
  `ranked_features.csv` 与已执行 PR-11 生成的 `final_review_summary.csv`；
- XGBoost fold preprocessing isolation：先建立单一 split base 时间轴
  （统一 resample、target 缺失处理、固定采样周期），随后对每个
  `train` / `gap_1` / `validation` / `gap_2` / `test` 分区独立执行 causal
  preprocessing 与 transform，lowpass / detrend / diff / forward-fill 状态
  不得跨 fold 边界；gap 继续承担 positive lag history buffer，且等于实际
  max used lag。

当前已实现（PR-13）：

- Web / API / CLI 正式双分支工作流总接入：
  - Web 与 CLI 只暴露 `raw` / `lowpass` / `lowpass_detrend` / `lowpass_diff`
    四种正式预处理模式；`lowpass_tau_minutes`（默认 5.0）与
    `diff_interval_minutes`（空值为 `None`）随配置传入并持久化到
    `run_config.json`；
  - `/api/analyze` 与 CLI `analyze` 使用 `run_initial_screening_workflow()`，
    不再以 `run_analysis()` 作为新初筛入口（旧入口保留兼容）；
  - 非 Raw 分析完成后进入 `awaiting_confirmation`，Web 展示冻结的
    `preprocessing_comparison.csv`（Raw vs Processed 对比），不发布正式 root
    初筛，前端不把任一分支伪装成正式排名；
  - 新增 `POST /api/confirm_initial_screening_branch` 与 CLI
    `confirm-branch`，只调用 `confirm_initial_screening_branch()`，不重新运行
    初筛、不重算 comparison、不复制 promotion；
  - downstream 开始前允许切换分支；`screening_downstream.lock` 创建后另一
    分支确认被禁用/拒绝（`initial_screening_branch_locked`），后端以正式
    runner 的 context gate 为最终权威；
  - 增强筛选 / 普通 Granger / RF/SHAP / 三级复核 / XGBoost 的 Web API 与
    CLI 命令统一改为调用 `run_*_for_active_branch()` 正式 runner，Web
    endpoint 不再保留并行 orchestration；
  - Result payload 增加 `preprocessingContext`、`preprocessingComparison`、
    `branchSelectionStatus`、`activeScreeningBranch`、
    `activePreprocessingMode`、`selectedPreprocessingMode`、`branchLocked`；
    `analysisContext.preprocess_mode` 表示 active 预处理模式，selected 模式
    单独保留；`awaiting_confirmation` 使用独立 pending payload，仅返回允许
    的对比/上下文下载；
  - 正式 branch 确定后，趋势图与 XY 散点矩阵使用 active preprocessing
    mode 及正式 `lowpass_tau_minutes` / `diff_interval_minutes` /
    `detrend_window`；pending 或尚未分析时仍仅按当前表单作为预览；
  - 下载白名单加入 `preprocessing_comparison.csv` 与
    `preprocessing_context.json`，`screening_branches/raw/*` 与
    `screening_branches/processed/*` 不开放任意路径下载；
  - LLM / 综合报告在正式分支确认前禁用（backend 保留
    `initial_screening_branch_not_confirmed` token）；
  - 旧二次验证重采样 / 白名单 override 从正式 Web 移除，正式 downstream
    只使用 active preprocessing context 与正式分析 `max_lag`。
