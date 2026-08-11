# Testing

## 测试原则

测试优先保护行为，而不是实现细节。

## 三类测试

### 行为测试

验证：

-   算法结果
-   排序行为
-   用户可见结果

### 契约测试

验证：

-   CSV/API 字段
-   数据流隔离
-   页面展示边界

### 源码约束测试

用于保护明确禁止模式：

-   后续结果进入初筛
-   闭环影响排序
-   XGBoost 回写初筛
-   abs(best_lag) 丢失方向

## 要求

不能通过删除测试、放宽断言或修改预期掩盖问题。

## 数据质量风险测试

数据质量测试覆盖以下行为：

-   普通质量超限（如 `missing_rate = 0.21`）只写入 `poor_data_quality`，
    仅提示并平滑降分：不封顶、不升级强风险计数、不强制
    `candidate_class = poor_quality`、不强制
    `recommended_use = poor_quality_variable`；
-   严重质量超限（`missing_rate > 0.50`、`saturation_ratio > 0.80`、
    `abnormal_jump_ratio > 0.05`、`robust_outlier_ratio > 0.05`）只写入
    `severe_data_quality`，作为一次强风险计数，`final_score` 上限为
    `0.44`，`risk_cap_reason = severe_data_quality`；
-   普通与严重标记互斥，严重条件不得同时出现 `poor_data_quality`；
-   严重阈值使用严格大于，恰好等于阈值不触发严重风险；
-   `data_quality_score` 连续衰减且四项为零时为 `1.0`；
-   `ranked_features.csv` 仍按 `final_score` 降序，CSV/API 字段结构不变。

历史兼容测试覆盖：`poor_data_quality` 与旧 `poor_quality` 分类、旧推荐用途或旧
`medium` 风险等级并存时，仍只作为普通质量提示；精确
`severe_data_quality` 才触发综合复核硬降级和 XGBoost 自动排除。排名基线的质量风险
及 A/B 风险计数按唯一变量断言，重复行不得重复计数。

## Raw 主筛查回归基线测试

使用项目内可控的小型固定数据（不依赖随机外部数据），锁定 Raw 主筛查行为：

- `final_score` 数值；
- `final_score` 降序排序；
- `driver_rank`；
- 初筛 Top-K；
- 候选池变量及来源；
- `best_lag` 的正负方向（`ranked_features.csv` 的 `lag` 列）；
- `ranked_features.csv` 关键字段；
- 新配置字段采用默认值时 Raw 输出不变。

浮点断言使用合理容差，但不得宽泛到掩盖评分变化。

## 预处理模式与配置字段测试

- `lowpass_tau_minutes <= 0` 被明确拒绝；
- `diff_interval_minutes <= 0` 被明确拒绝，`None` 合法；
- 四种契约模式（`raw`、`lowpass`、`lowpass_detrend`、`lowpass_diff`）可被配置对象
  表示；
- `transform_frame()` 支持 `lowpass*`；
- `transform_frame_causal()` 支持 `lowpass*`；
- causal 新模式不得使用未来样本；
- `analyze_numeric_frame()` 当前仍拒绝 `lowpass*`；
- Web/CLI 当前不得暴露 `lowpass*` 正式运行入口；
- Raw 和旧模式保持回归不变；
- `transform_frame()` 的基础能力不得自动进入评分、排序、候选池；
- 旧模式（`detrend`、`diff`、`detrend_diff`）保持兼容，且不被映射为新模式；
- 新增字段和模式不得进入评分、排序、候选池或默认执行流程。

## 单分支初筛 Runner 测试边界

- `run_initial_screening_branch()` 一次调用只执行一个分支，不自动执行另一分支；
- `branch=raw` 只允许 `preprocess_mode=raw`；
- `branch=processed` 只允许 `lowpass` / `lowpass_detrend` / `lowpass_diff`；
- 非法 branch 或 branch/mode 不匹配必须明确 `ValueError`，不得静默纠正；
- `raw` 分支输出隔离到 `screening_branches/raw/`，`processed` 分支输出隔离到
  `screening_branches/processed/`；
- 分支 runner 不得向运行根目录发布正式初筛文件；
- Raw 分支结果与正式 `analyze_numeric_frame()` / `run_analysis()` 的 Raw 结果
  一致；
- `lowpass_tau_minutes` 与 `diff_interval_minutes` 必须真实传入
  `transform_frame()`；
- 正式 `analyze_numeric_frame()` / `run_analysis()` 仍拒绝三个 `lowpass*` 模式；
- 两个分支输出互不覆盖，同分支重跑只覆盖自身分支；
- 不生成 `preprocessing_comparison.csv`、`preprocessing_context.json`；
- 不实现 confirmation / promotion。

## 双分支 comparison runner 测试边界

- `run_initial_screening_comparison()` 只允许 `lowpass` / `lowpass_detrend` /
  `lowpass_diff`，`raw` 与旧模式必须明确拒绝；
- 一次调用分别独立执行 raw 与 selected processed 两个分支，且不改写调用方
  `AnalysisConfig`；
- 两个分支仍输出到 `screening_branches/raw/` 与 `screening_branches/processed/`，
  不创建按 processed mode 命名的第三层目录；
- 双分支均成功后生成且只生成一个 `preprocessing_comparison.csv`（utf-8-sig），
  字段与顺序完全符合冻结契约；
- `variable` 为两个 `ranked_features` 的有序并集（raw 顺序 + 仅 processed 变量按
  processed 顺序）；
- `final_score_delta = processed - raw`，`rank_delta = raw_rank - processed_rank`；
- lag 保持正负方向，`lag_direction_changed` 按 negative/zero/positive 区分，
  任一侧缺失时保持缺失；
- Top-K 与 candidate membership 分别按各分支独立计算；
- risk 直接来自各分支 `risk_flags`，变量不存在侧保持缺失，不得用空字符串冒充
  无风险；comparison CSV 中缺失侧为 `NaN`，变量存在但无风险时为空字符串；
- comparison 不修改任何 branch 结果、不合并候选池、不产生推荐分支；
- 任一分支失败时不得生成半成品 comparison；
- 不生成 `preprocessing_context.json`，不向运行根目录发布正式初筛结果；
- 不实现 confirmation / promotion / Web / API / CLI 交互。
