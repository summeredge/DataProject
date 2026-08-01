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
