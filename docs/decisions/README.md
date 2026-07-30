# Decisions

ADR 用于记录长期有效设计决策。

只记录：

-   产品方向变化
-   架构变化
-   核心规则变化
-   明确禁止恢复的旧行为

状态：

-   Proposed
-   Accepted
-   Deprecated
-   Superseded

  编号   主题                       状态
  ------ -------------------------- ----------
  0001   初筛与后续分析隔离         Accepted
  0002   final_score 主导初筛排序   Accepted
  0003   闭环不影响初筛排序         Accepted
  0004   XGBoost 不回写初筛         Accepted

## 状态定义

- Proposed：提出但尚未生效
- Accepted：当前有效设计决策
- Deprecated：已废弃，不建议继续使用
- Superseded：已被新的 ADR 替代
