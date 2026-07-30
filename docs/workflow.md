# Workflow

## 默认流程

需求确认 → Codex 修改 → 测试 → Review → 用户确认

## 多 Agent

只有用户明确要求时使用：

Executor → Reviewer → 修复 → Verifier

禁止：

-   自动循环修复
-   自动提交
-   自动推送
-   自动合并

## Review

必须查看：

-   原始任务
-   完整 diff
-   测试结果
