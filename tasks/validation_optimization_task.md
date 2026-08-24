# DataProject 二级验证优化任务

## 总目标

将 DataProject 二级验证调整为：

-   候选验证层；
-   有限遗漏探索层。

保持项目核心约束：

-   `final_score` 不变；
-   `ranked_features.csv` 初筛排序不变；
-   Top-K 初筛顺序不变；
-   后续验证结果不得反向修改初筛结果；
-   不使用 `abs(lag)` 丢失滞后方向；
-   不新增第二套排序体系。

------------------------------------------------------------------------

# 执行规则

严格按照以下顺序执行：

-   Task-V1：统一二级验证结果层
-   Task-V2：二级验证页面重新设计
-   Task-V3：验证字段语义优化
-   Task-V4：模型发现模块重新定位
-   Task-V5：有限探索候选池
-   Task-V6：二级验证复核池
-   Task-V7：文档、契约和完整回归

每个 Task 完成后：

1.  停止进入下一 Task；
2.  执行测试；
3.  调用子agent @luna_worker检查 git diff；
4.  使用子agent @luna_worker检查是否存在 P0/P1；
5.  使用子agent @luna_worker检查是否破坏项目契约。

进入下一 Task 前必须满足以下关键门禁：

1. 当前 Task 相关测试通过；
2. 完整 `git diff`（含未跟踪文件）检查通过；
3. 由独立的 `@luna_worker` 完成 Review，确认无 P0/P1；
4. 初筛隔离、数据契约和核心排序约束检查通过；
5. 无当前 Task 引入的新回归。

已验证、已记录且明确与当前 Task 无关的既有 baseline 失败，可由用户授权
waiver，不阻塞进入下一 Task；该 waiver 不得覆盖当前 Task 相关测试失败、契约
破坏或 P0/P1 问题。最终发布仍需单独记录完整测试结果及所有遗留 waiver。

------------------------------------------------------------------------

# Task-V1：统一二级验证结果层

目标：

增加统一验证结论，不改变现有算法。

新增：

`validation_summary`

字段：

-   variable
-   validation_status
-   evidence_consistency
-   supporting_methods
-   limiting_factors

禁止：

-   新增 validation_score；
-   新增 validation_rank；
-   修改 final_score；
-   修改初筛排序。

------------------------------------------------------------------------

# Task-V2：二级验证页面优化

目标：

默认展示统一验证结论。

要求：

默认显示：

-   验证状态；
-   证据一致性；
-   主要支持证据；
-   限制因素。

详细结果折叠展示：

-   Enhanced Validation；
-   Granger；
-   Model Explanation。

只修改展示，不修改算法。

------------------------------------------------------------------------

# Task-V3：验证字段语义优化

目标：

区分不同阶段的 lag 和 model_lift。

增加：

-   initial_screening_lag
-   validation_lag
-   conditional_validation_lag
-   screening_model_lift
-   validation_model_lift

禁止：

-   abs(lag)；
-   合并不同阶段结果。

------------------------------------------------------------------------

# Task-V4：模型发现重新定位

目标：

模型发现作为遗漏探索，不作为验证结论。

要求：

-   不自动进入推荐；
-   不修改排序；
-   不改变 final_score。

------------------------------------------------------------------------

# Task-V5：有限探索候选池

目标：

建立有限遗漏检查。

规则：

探索范围：

`Rank K+1 ~ K+10`

例如 Top-K=20：

探索 Rank 21\~30。

限制：

-   最大输出 5 个发现候选；
-   不进行全量变量扫描；
-   不生成第二套排序。

------------------------------------------------------------------------

# Task-V6：二级验证复核池

目标：

在不改变一级初筛候选池语义的前提下，建立独立的二级验证复核池。

增加：

`verification_review_pool`

新增：

`verification_review_pool.csv`

字段：

-   variable
-   candidate_source
-   source_rank
-   include_reason

其中 `candidate_source` 仅表示进入二级验证复核池的来源，不得成为跨阶段通用字段。

取值：

-   initial_screening
-   manual_include
-   model_discovery

复核池来源：

-   初筛 Top-K：`initial_screening`；
-   人工加入：`manual_include`；
-   模型发现：`model_discovery`，必须经人工确认后加入。

允许影响：

-   Enhanced Validation；
-   Granger；
-   Model Explanation。

必须保持：

-   `recommended_candidates` 的既有候选来源语义；
-   `ranked_features.csv` 的初筛候选字段和排序；
-   初筛 Top-K；
-   `final_score`。

禁止影响：

-   final_score
-   ranked_features.csv
-   一级初筛候选池语义

------------------------------------------------------------------------

# Task-V7：文档和回归

更新：

-   docs/contracts.md
-   docs/architecture.md
-   docs/testing.md

增加测试：

-   后续验证不修改初筛；
-   模型发现不自动提权；
-   candidate_source正确；
-   lag方向保持；
-   缺失语义保持。

------------------------------------------------------------------------

# 最终验收

完成后确认：

一级：

-   final_score排序保持；
-   ranked_features.csv保持。

二级：

-   默认展示统一验证结论；
-   算法证据仍可展开。

探索：

-   只作为遗漏检查；
-   不成为第二套筛选系统。
