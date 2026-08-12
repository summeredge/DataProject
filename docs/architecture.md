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

## 预处理模式

正式 `analyze_numeric_frame()` / `run_analysis()` 当前实际执行模式：

```text
raw
detrend
diff
detrend_diff
```

预处理基础能力：

```text
transform_frame():
  支持 lowpass / lowpass_detrend / lowpass_diff

transform_frame_causal():
  支持 lowpass / lowpass_detrend / lowpass_diff
```

正式 `analyze_numeric_frame()` / `run_analysis()` 仍拒绝：

```text
lowpass
lowpass_detrend
lowpass_diff
```

模式语义与配置字段约束见 `docs/contracts.md`。`lowpass*` 模式可在配置对象中表示，
但正式分析入口执行前必须被明确拒绝，不得静默回退到 `raw`。

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
- branch 输出隔离到 `screening_branches/raw/` 或 `screening_branches/processed/`；
- branch runner 不向运行根目录发布正式初筛文件；
- 未锁定目录重新用于新 workflow 时先校验 mode，再清理旧 root 正式文件与旧
  context，避免暴露上一轮正式结果。

当前尚未实现：

- Web / API / CLI confirmation UI；
- conditional Granger / causal review / final review 正式 context 接入
  （PR-11）；
- XGBoost preprocessing consistency（PR-12）；
- Web / API / CLI 双分支交互。
