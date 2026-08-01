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
`final_score` 施加 `0.44` 上限。

仅 `severe_data_quality` 通过 `EVIDENCE_SCORE_CAPS` 将 `final_score`
封顶为 `0.44`，并在 `risk_cap_reason` 中记录 `severe_data_quality`。

`severe_data_quality` 是 `risk_flags` 内部标记，不新增 CSV/API 字段列。
