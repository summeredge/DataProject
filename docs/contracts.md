# Contracts

## 契约等级

### 稳定契约

未经确认不得修改：

-   文件名
-   字段名
-   字段语义
-   排序规则
-   API 输出结构

### 兼容契约

历史字段可以保留，但不得影响当前核心流程。

### 内部实现

允许调整，但不得改变外部行为。

## ranked_features.csv

用途：

第一阶段主筛查结果。

排序：

final_score 降序。

禁止：

-   后续模型回写
-   闭环状态影响排序
-   综合结果覆盖

## 数据质量风险语义

`data_quality_score` 仍通过连续系数直接参与初筛证据得分：

```text
evidence_confidence = data_quality_score
evidence_score = association_score × data_quality_score
```

普通 `poor_data_quality` 只写入 `risk_flags` 用于风险提示，不再对
`final_score` 施加 `0.44` 上限，也不属于强风险，不强制将候选分类为
`poor_quality` 或建议用途设为 `poor_quality_variable`。

仅 `severe_data_quality` 通过 `EVIDENCE_SCORE_CAPS` 将 `final_score`
封顶为 `0.44`，在 `risk_cap_reason` 中记录 `severe_data_quality`，并作为
强风险计数一次；只有它强制候选分类为 `poor_quality` 且建议用途为
`poor_quality_variable`。

`poor_data_quality` 与 `severe_data_quality` 互斥：达到严重阈值只写入
`severe_data_quality`，未达到严重阈值但达到普通阈值只写入
`poor_data_quality`。

两个标记都是 `risk_flags` 内部标记，不新增 CSV/API 字段列。

历史结果中，`poor_data_quality` 与旧的 `poor_quality` 分类、
`poor_quality_variable` 推荐用途或 `medium` 风险等级同时出现时，仍按普通质量提示
兼容处理；不得自动升级为严重风险。只有精确 `severe_data_quality` token 才会触发
下游硬降级和自动排除。

排名基线的质量风险数量及 A/B 风险数量均按当前 Top-N 内的唯一变量统计；同一变量的
重复行不重复计数。

## 预处理模式契约

新模式语义固定，不得修改：

```text
raw:
原始数据

lowpass:
原始数据 → 一阶低通

lowpass_detrend:
原始数据 → 一阶低通 → 去趋势

lowpass_diff:
原始数据 → 一阶低通 → 多点差分
```

旧模式继续保留兼容，且不得映射为新模式：

```text
detrend
diff
detrend_diff
```

约束：

- `detrend` 不得映射为 `lowpass_detrend`；
- `diff` 不得映射为 `lowpass_diff`；
- 旧模式的处理逻辑、旧配置文件和旧调用方兼容行为不得修改；

当前阶段：

1. `transform_frame()` 已支持：
   - `lowpass`
   - `lowpass_detrend`
   - `lowpass_diff`

2. `transform_frame_causal()` 已支持：
   - `lowpass`
   - `lowpass_detrend`
   - `lowpass_diff`

3. `analyze_numeric_frame()` / `run_analysis()`（legacy 正式入口）仍明确拒绝
   上述三个模式；正式初筛通过 `run_initial_screening_branch()` /
   `run_initial_screening_comparison()` / `run_initial_screening_workflow()`
   接入。

4. Web / API / CLI 已接入正式双分支工作流（PR-13）：`analyze` 统一使用
   `run_initial_screening_workflow()`；非 Raw 模式双分支完成后进入
   `awaiting_confirmation`，通过 `confirm_initial_screening_branch()` 人工
   确认正式分支后才允许 downstream。

5. 不得因为 `transform_frame()` / `transform_frame_causal()` 已具备基础能力，
   就绕过 raw + processed 双分支比较和人工确认流程。

6. 当前已具备单 branch 初筛执行能力：`run_initial_screening_branch()` 一次调用
   只执行一个分支（`raw` 或 `processed`），结果写入
   `screening_branches/raw/` 或 `screening_branches/processed/`，不发布到运行
   根目录。

7. 非 Raw 模式已具备双分支对比执行能力：`run_initial_screening_comparison()`
   对同一输入分别执行 raw 与 selected processed 两个独立分支，并生成
   `preprocessing_comparison.csv`；该入口只允许 `lowpass*`，不决定采用哪个
   分支，本身不承担 confirmation。

8. `run_initial_screening_workflow()` 已实现统一 workflow：
   - `raw` 只运行 raw 分支，事务性发布为正式 root 结果，状态为
     `not_required`，不需要人工确认；
   - `lowpass` / `lowpass_detrend` / `lowpass_diff` 复用
     `run_initial_screening_comparison()` 执行 raw + selected processed
     双分支并生成 `preprocessing_comparison.csv`，随后写入
     `preprocessing_context.json`，状态为 `awaiting_confirmation`；
   - 旧模式（`detrend`、`diff`、`detrend_diff`）必须被明确拒绝，且不得清理
     已有运行文件。

9. 已实现 `preprocessing_context.json`、人工 branch confirmation（复用已有
   branch 结果，禁止重新运行初筛）、事务 promotion（失败回滚）、downstream
   gate / lock 基础机制。

causal 组合固定为：

```text
lowpass_detrend
= lowpass → trailing detrend

lowpass_diff
= lowpass → historical multi-point diff
```

causal 路径全部只允许当前及历史数据，不得使用未来样本。

新模式不得自动进入评分、排序、候选池；非 Raw 正式初筛必须等待后续
双分支和确认机制。

## 预处理配置字段

分析配置对象包含以下字段：

```python
lowpass_tau_minutes: float = 5.0
diff_interval_minutes: float | None = None
```

字段语义：

- `lowpass_tau_minutes` 使用分钟，必须大于 `0`；
- `diff_interval_minutes` 使用分钟；
- `diff_interval_minutes = None` 表示自动采用一个分析采样周期；
- `0.0` 不得表示自动，必须大于 `0`；
- 本契约阶段不计算实际差分点数；
- 新字段不得影响 `raw` 或旧预处理模式的结果；
- 新字段不得自动进入评分、排序、候选池或默认执行流程。

## 初筛分支状态契约

分支选择状态：

```text
branch_selection_status:
  not_required
  awaiting_confirmation
  confirmed
```

字段及语义：

```text
selected_preprocessing_mode
active_screening_branch
active_preprocessing_mode
branch_selection_status
```

约束：

- 选择 Raw 时：

  - `branch_selection_status = not_required`
  - `active_screening_branch = raw`
  - `active_preprocessing_mode = raw`

- 非 Raw 双分支完成但未确认时：

  - `branch_selection_status = awaiting_confirmation`
  - `active_screening_branch` 缺失
  - `active_preprocessing_mode` 缺失

- 已确认时：

  - `branch_selection_status = confirmed`
  - `active_screening_branch` 为 `raw` 或 `processed`
  - `active_preprocessing_mode` 为实际进入后续阶段的模式

缺失值不得使用空字符串、`false` 或 `0.0` 代替。

当前阶段（PR-8）事实：

- `run_initial_screening_workflow()` 写入 `preprocessing_context.json`：
  - raw workflow 为 `not_required`，active branch/mode 均为 `raw`；
  - 非 Raw 双分支完成后为 `awaiting_confirmation`，active branch/mode 为
    JSON `null`；
- `confirm_initial_screening_branch()` 将状态更新为 `confirmed`：
  - 确认 `raw` 时 `active_preprocessing_mode = raw`；
  - 确认 `processed` 时 `active_preprocessing_mode` 为 selected 的
    `lowpass*` 模式；
  - `selected_preprocessing_mode` 始终为用户最初用于比较的模式，不得改写；
- 同 branch 重复确认是幂等 no-op；downstream 开始前允许切换到另一已存在
  branch；`screening_downstream.lock` 创建后禁止切换 branch；
- 缺失值必须真实写成 JSON `null`，不得使用 `""`、`false`、`0` 或 `0.0`
  代替。

## 分支产物目录契约

非 Raw 初筛运行的目标目录：

```text
run_directory/
├─ screening_branches/
│  ├─ raw/
│  └─ processed/
├─ preprocessing_comparison.csv
└─ preprocessing_context.json
```

约束：

- 两个分支分别生成独立的第一阶段产物；
- 未确认前不得在运行根目录发布正式初筛文件；
- 确认后只将选定分支发布为正式结果；
- 不得合并两个分支的候选池；
- 不得按变量选择两个分支中较高的分数；
- 不得生成新的综合评分。

当前阶段（PR-8）事实：

- 已具备单 branch 初筛执行能力，一次调用只执行一个分支，不自动执行另一分支；
- `raw` 分支输出隔离到 `screening_branches/raw/`，`processed` 分支输出隔离到
  `screening_branches/processed/`；
- 非 Raw 模式通过 `run_initial_screening_comparison()` 执行 raw + processed
  双分支，双分支均成功后生成 `preprocessing_comparison.csv`；
- 统一 workflow 通过 `run_initial_screening_workflow()` 完成状态闭环：raw
  自动发布，非 Raw 等待确认；
- `confirm_initial_screening_branch()` 只读取并验证已有 branch 文件后事务性
  promotion，确认 ≠ 重新运行初筛；
- promotion 前必须验证 9 个必需正式文件齐全，缺失时报
  `initial_screening_branch_output_incomplete`；
- promotion 使用 staging + backup + replace + context 更新的事务流程，任一
  步骤失败恢复原 root 文件与原 context，不会出现一半 Raw + 一半 Processed；
- 新 branch 缺少 `residual_corr_scores.csv` 时，旧 root 的可选文件残留必须
  被删除并进入 rollback 事务；
- `begin_downstream_stage()` 读取 context 作为 gate：`awaiting_confirmation`
  明确拒绝（`initial_screening_branch_not_confirmed`），`confirmed` /
  `not_required` 允许；首次通过后创建 `screening_downstream.lock`；
- lock 后禁止切换 branch，同 branch 确认为 no-op，lock 后再次启动 workflow
  必须拒绝（`initial_screening_run_locked`）；
- 未锁定的目录重新用于新 workflow 时，先校验 mode，再清理旧 root 正式文件
  与旧 context（raw 还清理旧 comparison）；
- 正式 `analyze_numeric_frame()` / `run_analysis()` 仍拒绝三个 `lowpass*` 模式。

## preprocessing_comparison.csv 契约

至少包含以下字段：

```text
variable
processed_mode
raw_available
processed_available
raw_final_score
processed_final_score
final_score_delta
raw_rank
processed_rank
rank_delta
raw_pearson
processed_pearson
raw_spearman
processed_spearman
raw_best_lag
processed_best_lag
lag_direction_changed
raw_in_top_k
processed_in_top_k
raw_candidate
processed_candidate
raw_risk_tags
processed_risk_tags
```

计算语义固定：

```text
final_score_delta =
processed_final_score - raw_final_score
```

```text
rank_delta =
raw_rank - processed_rank
```

因此 `rank_delta > 0` 表示预处理后排名提升。

其他要求：

- 变量集合使用两个分支的并集；
- 单侧不存在的字段保持缺失；
- 任一分支无有效滞后时，`lag_direction_changed` 保持缺失；
- 不得将缺失写成 `false`；
- 不得使用 `abs(lag)` 或 `abs(best_lag)` 判断方向；
- 对比文件只用于展示差异，不参与评分、排序或候选生成。

当前阶段（PR-7）：该文件由 `run_initial_screening_comparison()` 在 raw 与
processed 两个分支均成功后生成，编码沿用项目 CSV 约定（utf-8-sig）；字段顺序、
缺失语义与 delta 计算规则按上文冻结；缺失侧单元格以 `NaN` 明确标记，变量存在但
无风险时 `*_risk_tags` 保持空字符串，两者在文件中可区分；不得新增推荐分支或综合
评分字段。

## preprocessing_context.json 契约

至少包含以下字段：

```text
selected_preprocessing_mode
active_screening_branch
active_preprocessing_mode
lowpass_tau_minutes
requested_diff_interval_minutes
effective_diff_points
effective_diff_interval_minutes
resample_rule
branch_selection_status
```

要求：

- 不适用或尚未计算的字段使用缺失语义；
- 不得使用 `0.0` 代替缺失；
- `selected_preprocessing_mode` 表示用户选择用于比较的模式；
- `active_preprocessing_mode` 表示正式进入后续阶段的模式；
- 两者不得混为一个字段。

字段语义（PR-8 冻结）：

- `lowpass_tau_minutes`：`raw` 为 `null`；`lowpass*` 记录
  `config.lowpass_tau_minutes`；
- `requested_diff_interval_minutes`：仅 `lowpass_diff` 有值，记录用户配置
  （`None` 时保持 `null`）；
- `effective_diff_points` / `effective_diff_interval_minutes`：仅
  `lowpass_diff` 计算，必须复用 `resolve_diff_interval()` 及现有采样周期 /
  resample 规则，不得重新实现 round / sampling interval / effective points
  数学规则；
- `resample_rule`：记录真实配置，无配置为 `null`；
- 不适用、尚未计算、尚未确认的字段统一使用 JSON `null`，禁止用 `0`、
  `0.0`、`false`、`""` 代替缺失；真实有效 `0.0` 不得被转换为缺失。

当前阶段（PR-8）已生成该文件，写入方式为临时文件 → 完整写入 →
`os.replace`，编码 UTF-8（`ensure_ascii=False`、`indent=2`），不得引入第三
方依赖；不得新增评分、排序或自动推荐字段。

## 后续阶段正式 branch/context 契约（PR-9 / PR-10 / PR-11）

增强筛选、普通 Granger、RF/SHAP 模型与三级复核（conditional Granger /
causal review / final review）的正式 backend 入口（`pipeline.py`）：

```python
prepare_downstream_analysis_context(run_dir, ...)
run_enhanced_screening_for_active_branch(run_dir, ...)
run_granger_for_active_branch(run_dir, ...)
run_model_for_active_branch(run_dir, ...)
run_causal_review_for_active_branch(run_dir, ...)
```

执行顺序固定：

```text
读取并验证 preprocessing_context.json
→ 验证正式 root 初筛输入
→ 解析 active preprocessing 配置
→ begin_downstream_stage(run_dir)（首次通过创建 screening_downstream.lock）
→ causal preprocessing / active causal transform / target mask / standardize
→ 调用现有增强筛选 / 普通 Granger / 模型 service
→ 写出现有阶段结果文件
```

约束：

- 后续阶段只能消费正式 root 的 `ranked_features.csv` /
  `recommended_candidates.csv`、`risk_flags.csv` 等已 promotion 初筛文件；
  禁止读取
  `screening_branches/raw/`、`screening_branches/processed/` 或
  `preprocessing_comparison.csv` 构造候选，禁止 Raw ∪ Processed 合并；
- 预处理参数以 context 为准：`active_preprocessing_mode` 覆盖调用方
  `config.preprocess_mode`；`lowpass_tau_minutes` 来自
  `context.lowpass_tau_minutes`；`diff_interval_minutes` 来自
  `context.requested_diff_interval_minutes`（`null` 时沿用现有
  `None` 语义）；`resample_rule` 来自 `context.resample_rule`；
  `selected_preprocessing_mode` 只表示用户最初选择，不得覆盖
  `active_preprocessing_mode`；
- 构造正式 downstream config 必须通过 `dataclasses.replace`，不得原地修改
  调用方 config；
- 增强筛选、普通 Granger、RF/SHAP 与 conditional Granger/三级复核必须
  使用 causal preprocessing；完整时间轴先变换，工况只作为 target mask，
  standardization 仍以 target mask 拟合。初筛的历史筛查可继续使用回顾性
  transform，XGBoost 继续使用独立的 fold-safe causal preprocessing；
- `branch_selection_status = awaiting_confirmation` 时所有正式入口必须拒绝：
  `initial_screening_branch_not_confirmed`；context 缺失 / 非法：
  `initial_screening_context_missing` / `initial_screening_context_invalid`；
- 正式 root 缺少必要初筛输入（如 `ranked_features.csv`、
  `recommended_candidates.csv`；模型还需 `risk_flags.csv`）时必须失败：
  `initial_screening_formal_output_missing`，且该检查在创建 downstream lock
  之前完成，不得回退到 branch 目录；
- 第一个成功进入 downstream 的 stage 创建 `screening_downstream.lock`；
  lock 已存在表示 branch 已冻结，不得阻止其他后续 stage；
- 增强筛选 / Granger / 模型不得改写正式初筛文件：
  `ranked_features.csv`、`recommended_candidates.csv`、
  `causal_review_candidates.csv` 在运行前后必须保持一致；
- 阶段输出沿用现有契约：增强筛选写
  `model_lift_scores.csv`、`rolling_corr_scores.csv`、
  `enhanced_validation_summary.csv`（及 `secondary_candidate_context.csv`），
  普通 Granger 写 `granger_tests.csv`，RF/SHAP 模型写
  `shap_or_importance.csv`、`model_variable_importance.csv`、
  `model_discovered_candidates.csv`，不得新增综合评分 CSV；
- 模型复用现有 `fit_explainable_model()` /
  `build_model_variable_importance()` / `build_model_discovered_candidates()`，
  不改 RandomForest 参数与 SHAP fallback；SHAP 不可用时沿用
  `random_forest_feature_importance`，不得强制 SHAP 或使阶段失败；
- 模型输出仅作为解释/复核辅助（`model explanation only; not a causal
  conclusion`），不得回写初筛，不得表述为初筛结果、确定性因果或已确认根因；
- 普通 Granger 保留 signed lag，不得使用 `abs(lag)` / `abs(best_lag)` /
  `abs(granger_lag)` 决定时间方向；输出不得表述为确定性因果；
- PR-9 接入 ordinary/bivariate Granger，PR-10 接入 RF/SHAP/model
  discovery，PR-11 接入 conditional Granger / causal review / final review；
  XGBoost（PR-12）接入 fold-safe 正式 runner；Web/API/CLI 正式总接入
  （PR-13）只改变入口 orchestration，不改变上述后端契约。

趋势图与 XY 散点矩阵继续使用历史观察用途的回顾性 `transform_frame()`。
正式 branch 已确定时，query 必须使用 active mode 及正式
`lowpass_tau_minutes` / `diff_interval_minutes` / `detrend_window`；用户之后
修改表单但未重新分析时不得覆盖 active context。`awaiting_confirmation` 或
尚未分析时只允许按当前表单作为预览，不得据此确认 branch。

## 三级复核正式 branch/context 契约（PR-11）

三级复核正式 backend 入口：

```python
run_causal_review_for_active_branch(
    run_dir,
    *,
    base_config=None,
    control_columns=None,
    maxlag=None,
    min_rows=60,
    top_n=None,
    conditional_lag_mode="ranked_window",
    conditional_lag_window=5,
    conditional_fallback_maxlag=24,
    conditional_baseline_maxlag=24,
    progress_callback=None,
)
```

执行顺序固定：

```text
读取并验证 preprocessing_context.json
→ 验证正式 root 初筛输入
→ 根据 active preprocessing 构造 downstream config
→ begin_downstream_stage(run_dir)（首次通过创建 screening_downstream.lock）
→ 读取正式三级复核候选
→ 准备 active preprocessing 数据
→ 调用 run_causal_review_stage()
→ 写三级复核结果
```

约束：

- 必需正式 root 输入固定为 `ranked_features.csv`、
  `recommended_candidates.csv`、`causal_review_candidates.csv`、
  `risk_flags.csv`；任一缺失报
  `initial_screening_formal_output_missing`，不得回退读取
  `screening_branches/` 补救；
- 正式三级候选唯一来自 root `causal_review_candidates.csv`，不得重新调用
  `build_causal_review_candidates()`，不得通过 `ranked_features.csv`、
  `secondary_candidate_context.csv`、`model_discovered_candidates.csv` 或
  `preprocessing_comparison.csv` 重新生成 / 扩大候选池；非 active branch
  的候选不得进入；
- 执行前后 `causal_review_candidates.csv` 必须 byte-identical，正式 runner
  不得写入该文件；
- `top_n=None`（默认）时把正式候选全量传入现有 `run_causal_review_stage()`，
  不得重新加入 `final_score >= 0.30`、candidate grade、review tier、
  risk level、model importance 或 Granger significance 硬裁剪；显式
  `top_n` 只沿用 stage 现有 `head(top_n)` 语义，不得重新排序后再截断；
- active preprocessing 必须来自 context：确认 Raw 时一律使用 `raw`（即使
  `selected_preprocessing_mode` 为 `lowpass*`）；确认 Processed 时使用
  context 的 `preprocess_mode`、`lowpass_tau_minutes`、
  `requested_diff_interval_minutes`、`resample_rule`，调用方 config 不得
  覆盖；
- 复用统一 causal secondary frame helper / `_target_segment_mask()`；
  显式 `control_columns` 时以
  causal helper 的 `protected_columns=resolved_control_columns`
  保护控制列，不修改其算法；
- control columns 解析保持现有三级复核行为：显式 `control_columns` → 否则
  `config.residual_control_columns` → 否则 `config.capacity_columns` → 否则
  `[]`；显式空列表保持“无控制列”语义，不得回退 config；必要时复用现有
  excluded-column 校验 helper，不得自动识别新的控制变量；
- `maxlag=None` 时使用现有 `config.resolved_granger_maxlag()`，不新增
  maxlag 推断算法；`min_rows` 与 conditional 参数默认值与现有 stage 一致并
  完整透传；
- signed lag 保持方向：正式 runner 使用 `ranked_features.csv` 的真实
  signed lag，不得使用 `abs(lag)` / `abs(best_lag)` / `abs(granger_lag)`
  把负 lag 转正；
- 成功执行仅写现有四个三级输出：`conditional_granger_scores.csv`、
  `causal_review_report.csv`、`causal_review_evidence.csv`、
  `final_review_summary.csv`，字段与 schema 保持现有实现，不得新增
  `causal_score.csv` / `combined_score.csv` / `final_rank.csv` 等综合评分
  文件；
- optional evidence（`enhanced_validation_summary.csv`、
  `granger_tests.csv`、`model_variable_importance.csv`）沿用现有 stage
  从 `output_dir` 可选读取：存在 → 作为已执行辅助证据；缺失 → 保持
  缺失/未执行语义，不得自动运行对应阶段，不得用 `0.0` /
  `unsupported` / negative evidence 冒充“未执行”；
- 三级复核不得改写任何正式初筛文件：执行前后 `ranked_features.csv` /
  `recommended_candidates.csv` / `causal_review_candidates.csv` /
  `risk_flags.csv` byte-identical，`final_score` / `driver_rank` / Top-K /
  初筛推荐顺序不变；final review 只是后续复核结果，不成为新的“初筛排名”；
- 三级复核作为第一个 downstream stage 时创建 `screening_downstream.lock`；
  已有 lock（Enhanced / Granger / Model 先运行）后仍可执行，不得报
  `initial_screening_run_locked`；三级复核运行后 branch 不可切换
  （`initial_screening_branch_locked`）；
- `awaiting_confirmation` 明确拒绝 `initial_screening_branch_not_confirmed`，
  context 缺失 / 非法保持
  `initial_screening_context_missing` / `initial_screening_context_invalid`，
  均不得 fallback；
- 三级复核不自动调用增强筛选、普通 Granger、模型或 XGBoost 阶段，阶段保持
  独立；XGBoost 正式 branch/context 与 fold preprocessing isolation
  （PR-12）已实现，Web/API/CLI 双分支工作流总接入（PR-13）已完成。

## XGBoost 正式 fold-safe 有效样本与审计字段契约（PR-12）

- split-base 的 `train >= 100` / `validation >= 30` / `test >= 30` 仅为初始
  fold 几何约束；正式 fold-safe XGB 在每个 fold 完成 preprocessing、工况
  target mask、lag feature alignment 与 complete-case dropna 后，再次按同一
  下限检查真正送入模型的有效行数。
- 任一 partition 有效样本不足时整个 XGB 返回 `invalid_input`，不得静默跳过
  fold、减少 fold 数、自动放宽下限、关闭工况 mask、减少 lag 或继续生成部分
  结果；已有 `xgb_validation/` 五个输出按 transactional 行为保护。
- formal fold-safe `row_count` 表示实际 out-of-time prediction rows（即
  `xgb_predictions.csv` 行数）；legacy runner 的 `row_count` 仍表示 legacy
  feature-set 有效行数。字段名不变。
- `data_fingerprint` 覆盖所有 fold 实际用于模型的 train / validation / test
  输入（fold id、partition 类型、时间索引、target、M1 与 M2 特征），不再只
  覆盖第一个 fold 的 train M1；不得包含文件路径、`run_dir`、`created_at` 或
  随机值，相同输入重复执行必须稳定。

## Web / API / CLI 正式接入契约（PR-13）

### 正式入口

- Web `/api/analyze` 与 CLI `analyze` 使用 `run_initial_screening_workflow()`，
  不再以 `run_analysis()` 作为新初筛入口；`run_analysis()` 保留内部/历史
  兼容，不删除。
- Web/CLI 正式预处理模式固定为 `raw` / `lowpass` / `lowpass_detrend` /
  `lowpass_diff`；旧模式（`detrend` / `diff` / `detrend_diff`）继续保留
  backend compatibility，但不出现在正式选择中，也不得静默映射为新模式。
- `lowpass_tau_minutes` 默认 `5.0`，仅 `lowpass*` 生效，Raw 时不参与实际
  分析；`diff_interval_minutes` 仅 `lowpass_diff` 生效，空值语义为 `None`
  （一个分析采样周期），非空必须大于 `0`，不得用 `0` / `0.0` / `""` 代替
  `None`。
- `run_config.json` 持久化并恢复 `preprocess_mode`、
  `lowpass_tau_minutes`、`diff_interval_minutes`、`resample_rule`、
  `detrend_window`。

### Raw 工作流

- 只运行 Raw 分支，不运行 processed，不生成
  `preprocessing_comparison.csv`；自动 promotion，状态
  `not_required`，可直接进入 downstream，不增加多余的“确认 Raw”步骤。

### 非 Raw 工作流

- 同时运行 Raw + 所选预处理模式两个独立初筛，生成
  `preprocessing_comparison.csv`；状态 `awaiting_confirmation`，正式 root
  初筛文件（`ranked_features.csv`、`recommended_candidates.csv`、
  `causal_review_candidates.csv`、`summary.md` 等）不得存在。
- Web 必须显示 Raw vs Processed 对比，不得把任一分支伪装为正式排名 /
  Top-K / 推荐变量；不得自动选择“更优”分支，不得按 `final_score` 判断分支，
  不得合并 Raw/Processed 候选。
- 人工确认通过 `POST /api/confirm_initial_screening_branch`（参数
  `run_id` + `branch = raw | processed`）或 CLI `confirm-branch`，backend
  只调用 `confirm_initial_screening_branch()`；确认 ≠ 重新运行初筛，不得
  重算 comparison，不得由前端复制文件。
- 确认成功后重新读取 root 正式结果并同步刷新
  `rankedFeatures` / `recommendedCandidates` / `riskFlags` / `overview` /
  `downloads` / `analysisContext`；downstream 开始前允许切换分支，创建
  `screening_downstream.lock` 后禁止切换
  （`initial_screening_branch_locked`）。

### Result payload

正式 API 返回增加（数据直接来自 `preprocessing_context.json`、
`preprocessing_comparison.csv`、`screening_downstream.lock`）：

```text
preprocessingContext
preprocessingComparison
branchSelectionStatus
activeScreeningBranch
activePreprocessingMode
selectedPreprocessingMode
branchLocked
```

`analysisContext.preprocess_mode` 表示 active 预处理模式（例如
`selected = lowpass_diff`、`confirmed = raw` 时显示 `raw`）；
`selectedPreprocessingMode` 保留最初比较模式，不得混淆两者。

`awaiting_confirmation` 使用独立 pending payload，只允许返回 `run_id`、
`preprocessingContext`、`preprocessingComparison`、`branchSelectionStatus`、
`selectedPreprocessingMode`、`activeScreeningBranch = null`、
`activePreprocessingMode = null`、`downloads`（仅真实存在且允许下载的
comparison/context 文件）以及必要任务时间信息；不得把
`screening_branches/raw/*` 或 `screening_branches/processed/*` 包装为正式
`rankedFeatures`。

### Downstream 正式 runner

- Web `/api/run_enhanced_screening`、`/api/run_granger`、`/api/run_model`、
  `/api/run_causal_review`、`/api/run_xgb_validation` 与 CLI
  `run-enhanced` / `run-granger` / `run-model` / `run-causal-review` /
  `run-xgb` 统一调用 `run_*_for_active_branch()` 正式 runner，由
  `prepare_downstream_analysis_context()` / `begin_downstream_stage()` 执行
  最终检查；Web endpoint 不复制 context state machine。
- `awaiting_confirmation` 时所有正式 downstream 入口明确阻断
  （`initial_screening_branch_not_confirmed`），不得自动 Raw fallback。
- 旧二次验证参数（`secondary_resample_mode` / `secondary_resample_rule` /
  `secondary_max_lag`）从正式 Web 二次验证配置区移除；
  `secondary_include_variables` 不再通过旧 endpoint 扩展正式候选，正式
  downstream 不再提供 Raw ∪ Processed ∪ secondary whitelist 入口。
- `risk_flag_filter` 仅作为结果展示过滤，不改变
  `causal_review_candidates.csv` 与 conditional Granger 输入候选集。
- 下载白名单加入 `preprocessing_comparison.csv` 与
  `preprocessing_context.json`；`screening_branches/raw/*`、
  `screening_branches/processed/*` 与任意路径下载继续拒绝。
- LLM Prompt / 综合报告在正式分支确认前不启用（保留
  `initial_screening_branch_not_confirmed` 等后端 token）。
- 旧 CLI `--enable-granger` / `--enable-model`：Raw 模式可在正式初筛
  promotion 完成后调用对应正式 runner；非 Raw 模式明确提示先
  `confirm-branch`，不得自动选择 Raw 或 Processed。
