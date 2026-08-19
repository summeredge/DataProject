# Learnings

## [LRN-20260818-001] task_review

**Priority**: high
**Status**: resolved
**Area**: tests

### 内容
将 Shadow 评分晋级为正式评分时，仅复用公式单元测试不足以证明集成接线完整。本次首次接线
只把 Regime 传入正式评分而遗漏 Rolling；独立诊断仍包含两者，所以普通契约测试和 A/B
汇总都可能掩盖该差异。

### 建议修复
晋级后在真实数据或端到端场景中，按变量断言正式 `final_score` 与同一冻结输入生成的显式
评分分解完全一致，同时单独验证公开 sidecar/output 边界未改变。

### 元数据
- Source: task_review
- Pattern-Key: shadow-promotion-formal-diagnostic-equivalence

---
