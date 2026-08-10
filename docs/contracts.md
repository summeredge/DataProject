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

2. `transform_frame_causal()` 尚未支持上述三个模式，必须明确拒绝。

3. `analyze_numeric_frame()` / 正式初筛流程尚未接入上述三个模式，必须明确拒绝。

4. Web / CLI / API / 双分支正式运行尚未接入。

5. 不得因为 `transform_frame()` 已具备基础能力，就绕过 raw + processed
   双分支比较和人工确认流程。

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

当前阶段只定义状态契约，不实现状态流转。

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

当前阶段不创建这些目录或文件。

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

当前阶段不生成该文件。

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

当前阶段不生成该文件。
