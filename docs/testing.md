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

-   普通质量超限（如 `missing_rate = 0.21`）仅提示并平滑降分，不封顶；
-   严重质量超限（`missing_rate > 0.50`、`saturation_ratio > 0.80`、
    `abnormal_jump_ratio > 0.05`、`robust_outlier_ratio > 0.05`）触发
    `severe_data_quality`，`final_score` 上限为 `0.44`；
-   严重阈值使用严格大于，恰好等于阈值不触发严重风险；
-   `data_quality_score` 连续衰减且四项为零时为 `1.0`；
-   `ranked_features.csv` 仍按 `final_score` 降序，CSV/API 字段结构不变。
