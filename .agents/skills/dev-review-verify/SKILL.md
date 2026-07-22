---
name: dev-review-verify
description: 仅当用户明确要求使用 dev-review-verify 工作流，或明确要求执行  「独立 Review → 最多一次自动修复 → Verifier 验证」时使用。普通代码修改、Bug 修复、UI 调整、参数修改、测试补充、代码解释和一般重构均不得自动使用本 Skill。用户未明确要求该审查流程时，由当前 Codex 主 Agent 按普通方式直接处理。
---

# Dev Review Verify

根线程承担 Planner：接收用户或 ChatGPT 提供的任务，明确验收标准，并只调度一级子 Agent `executor`、`reviewer`、`verifier`。子 Agent 不得继续创建 Agent。不得自动进入下一任务、提交或推送；所有最终结果都必须停止并等待用户决定。

## 阶段 1：执行

调用 `executor` 完成明确的代码修改、必要测试与测试执行。保留其 `EXECUTION_STATUS`、`CHANGED_FILES`、`TEST_RESULTS` 与 `UNRESOLVED_ITEMS` 输出，供后续阶段使用。

## 阶段 2：Review

调用 `reviewer` 进行独立审查。向其提供原始任务、验收标准、当前完整 git diff、涉及的完整代码及测试结果，而非仅提供 Executor 的描述。

本流程最多调用 Reviewer 两次：第一次为完整 Review；如需第二次，只可验证修复对应的 Findings。

## 阶段 3：结果处理

若 `REVIEW_STATUS = PASS`：

1. 输出 Executor 的执行结果；
2. 输出 Reviewer 的 Review 结果；
3. 停止流程并等待用户决定。

若 `REVIEW_STATUS = ISSUES_FOUND`：

1. 将完整 Findings 发送给 `executor`；
2. `executor` 仅可自动修复一次，且只修复明确 Findings；
3. 收到修复内容与重新测试结果后，进入阶段 4。

若 `REVIEW_STATUS = BLOCKED`：输出阻塞原因，停止流程并等待用户决定。

不得因同一 Findings 再次调度修复；不得形成循环。

## 阶段 4：验证

调用 `verifier`，并提供原始任务、Reviewer Findings、Executor 的修复内容、当前完整 git diff 和修复后的测试结果。`verifier` 只验证修复内容，不扩大为第二次完整 Review。

无论 `VERIFICATION_STATUS` 为 `PASS`、`FAIL` 或 `BLOCKED`：输出最终报告，停止流程，并等待用户决定。不得自动开始下一项工作或进行提交、推送。
