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
