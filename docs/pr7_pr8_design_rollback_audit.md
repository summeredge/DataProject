# PR-7 / PR-8 设计回退审计

## 审计范围与基线

本审计以当前 `main` 的提交 `9bd433b1319689f7e7a2ce43a1a1f2d66800f351` 为工作起点。当前 `main` 与 `origin/main` 均指向该提交；本审计只将该提交作为历史比较起点，不把接手时已有的未提交任务改动误记为干净工作树。

按任务指定的历史检查：

- `b125f6cb2e7d44735e1f72902a1710bb96b83a7a` 的直接父提交为 `c25da7308368c41eb8e2b3bb2f0bfbb84d17f294`。
- `b125f6c` 的实际 patch 首次把自动闭环风险上下文接入 `screening._finalize_driver_ranking`，并新增闭环融合配置、接口和输出。
- 因此本任务的 `PRE_PR7_BASE` 记录为 `c25da7308368c41eb8e2b3bb2f0bfbb84d17f294`。对于更早的初筛排序变更，另以 `0e77b41` 的实际 patch 作为来源审计，不以提交标题代替行为证据。

审计差异范围为 `PRE_PR7_BASE..9bd433b`，并补充检查了 `0e77b41`、`9592267`、`4f01107` 等 PR-7/PR-8 相关 patch 及其父版本，用于识别已经被后续提交交织或清理的行为。

## 改动处理清单

| 当前行为或改动 | 来源 | 处理决定 | 理由 | 对应测试 |
|---|---|---|---|---|
| `PRIMARY_RANK_COLUMN = "driver_rank"`，初筛结果按 `driver_rank` 升序返回 | `0e77b41` PR-7 | 撤销 | 原始初筛目标是统计候选的稳健综合得分排序；工程类别系数不能改变统计候选顺序 | `test_initial_screening_orders_candidates_by_final_score_descending`、`test_initial_screening_top_k_uses_final_score_not_driver_priority_score` |
| `driver_priority_score = final_score * driver_priority_factor` 参与主排序、Top-K 和 Web 默认排序 | `0e77b41`、PR-8 后续排序契约 | 撤销主行为；兼容字段可保留但不得消费 | 驱动优先级是后续工程解释概念，不是初步统计筛选分数 | `test_initial_recommendations_preserve_final_score_order`、`test_compatibility_priority_fields_do_not_change_initial_results` |
| `candidate_class` 选择 `driver_priority_factor` 并影响 `driver_rank` | PR-7/PR-8 | 撤销对初筛排序、Top-K、推荐的影响 | 候选类别不应二次修正初步统计候选 | `test_compatibility_priority_fields_do_not_change_initial_results` |
| 闭环风险上下文、自动闭环阈值/系数和闭环融合输出 | `b125f6c` 及后续闭环链路 | 撤销初筛消费者；无有效消费者的专用代码删除 | 闭环判断不能改变初筛评分、等级、排序、建议用途或普通推荐集合 | `test_closed_loop_legacy_inputs_do_not_change_initial_results`、`test_closed_loop_fields_are_not_initial_screening_consumers` |
| 人工闭环字段 `manual_closed_loop_variables`、`manual_non_closed_loop_variables` 及历史状态字段 | PR-1～PR-7 兼容遗留 | 惰性兼容读取；不进入当前初筛输入或输出 | 允许历史配置读取但避免旧产品行为复活 | `test_closed_loop_legacy_inputs_do_not_change_initial_results` |
| `layer1_association_status`、`layer2_temporal_status`、`layer3_independence_status`、`layer4_model_status` 进入主输出 | `1de0b66`、PR-8 | 撤销初步分析展示和初步 API 输出 | 四层状态依赖后续证据，不能在后续分析未执行时提前出现在初筛 | `test_initial_api_filters_four_layer_fields`、`test_initial_web_contract_excludes_four_layer_fields` |
| `four_layer_coverage_status`、`four_layer_missing_items` | `e72f805` PR-8 | 撤销初步分析展示；后续页面按真实结果文件独立展示 | 覆盖状态不是初步综合评分，也不能把未运行阶段包装成结论 | `test_initial_api_filters_four_layer_fields`、`test_initial_web_contract_excludes_four_layer_fields` |
| `evidence_support_items`、`evidence_against_items`、`evidence_conflict_items`、`candidate_summary` | `evidence_explanations.py`、`e72f805`、`9bd433b` | 删除其初步路径调用和初筛展示；确认无有效消费者后删除专用模块及测试 | 这些是四层解释文案，不属于初步分析真实产物 | `test_initial_screening_source_has_no_four_layer_explanation_call` |
| 缺失证据与真实 `0.0` 分离 | `4f01107` 后的正确性修复、PR-8C | 保留 | 未计算/不可用不等于弱证据；真实零值仍应作为已计算的弱证据 | `test_missing_prediction_is_not_treated_as_zero`、`test_real_zero_optional_evidence_is_valid` |
| 可用权重重新归一化 | PR-8C | 保留 | 缺失可选组件不应被零值惩罚，也不能因填零造成错误权重语义 | `test_profile_score_renormalizes_only_available_component_weights` |
| 滞后正负方向、目标领先/变量领先、边界语义 | PR-8C 及时间轴修复 | 保留 | 初筛必须保持物理时间语义；禁止恢复 `abs(best_lag)` | `test_service_metrics`、`test_initial_screening_synthetic_baseline`、源码契约 `test_statistical_screening_contracts_remain_in_source` |
| 数据质量的实际维度几何聚合、缺失/异常/饱和/跳变处理 | PR-8C | 保留 | 属于统计正确性和数据质量结果，不是产品方向行为 | `test_data_quality_smooth`、`test_pr_8c_source_contracts_cover_each_repaired_ranking_path` |
| 冗余代理连通分组、列顺序不变性和无法区分时的保守处理 | PR-8C | 保留 | 避免把代理关系误判成已识别的具体驱动 | `test_initial_screening_synthetic_baseline` 中代理场景、`test_redundancy_resolution_uses_evidence_not_column_order` |
| 确定性合成场景：真实滞后、下游响应、共同驱动、共线代理、非线性、工况反转、异常点、滞后边界、纯噪声 | PR-8 合成基准 | 保留并改验初筛契约 | 这些场景可验证统计正确性和伪相关防护，但不应冻结四层 UI 字段 | `test_initial_screening_synthetic_baseline` 及其初筛契约更新 |
| 版本化本地 fixture、确定性随机种子、无 Git 历史依赖 | PR-8 测试基础设施 | 保留 | 正常测试必须直接使用仓库 fixture，不依赖提交历史 | `test_committed_initial_baseline_matches_actual_report` |
| 初步分析完成后立即显示增强筛选/滚动稳定性/工况稳健性/增量验证结论 | 当前 `service.analyze_numeric_frame`、`_build_result_payload`、初筛 Web 表格 | 撤销初步路径消费；后续结果只由对应 API/结果文件提供 | 未运行后续阶段时初筛不显示后续字段、状态或结论 | `test_initial_analysis_does_not_expose_unexecuted_followup_results`、`test_initial_service_does_not_execute_followup_analyses` |
| `/api/analyze`、`/api/run_enhanced_screening`、`/api/run_granger`、`/api/run_model`、`/api/run_causal_review`、`/api/run_xgb_validation` 混合到同一初步结果表 | 当前 Web API 结果加载与渲染链 | 保留路由独立性，按阶段隔离 payload、文件和页面 | 后续增强分析、Granger、模型分析、综合复核必须可独立运行 | `test_initial_analysis_does_not_expose_unexecuted_followup_results`、各现有对应路由测试 |
| `summary.md` 使用“四层工业时序筛查摘要”“四层证据解释”“驱动因素候选排序” | PR-8 报告改动 | 撤销，恢复初步筛选摘要标题和术语 | 报告应定位为初步统计筛选，不提前宣称四层证据或驱动因素判断 | `test_summary_restores_initial_screening_positioning` |

## 阶段边界表

| 用户阶段 | 入口/结果文件 | 允许展示的数据 | 未执行时的表示 |
|---|---|---|---|
| 初步分析 | `/api/analyze`、`ranked_features.csv`、`recommended_candidates.csv`、初步 `summary.md` | 初始 Pearson/Spearman、主导相关方法、相关方向、最佳滞后、时间关系、`final_score` 稳健综合得分、基础风险、数据质量、样本数、当前阶段建议用途 | 后续字段不进入初步主表/API；缺失字段不伪造为状态 |
| 增强筛选 | `/api/run_enhanced_screening`、`enhanced_validation_summary.csv`、增强结果文件 | 滚动稳定性、工况稳健性、增量验证及增强阶段的真实结果 | `未执行` / `未计算` / 无对应结果文件 |
| Granger/条件分析 | `/api/run_granger` 及 Granger/条件 Granger 结果文件 | 对应运行产生的预测时序关系、滞后和显著性结果 | `未执行` / `未计算` / 无对应结果文件 |
| 模型分析 | `/api/run_model`、模型重要性/发现结果文件 | 对应运行产生的模型特征、重要性和模型解释 | `未执行` / `未计算` / 无对应结果文件 |
| 综合复核 | `/api/run_causal_review`、`final_review_summary.csv` | 只汇总已经实际执行并生成文件的阶段结果 | 不生成支持/未支持/证据不足结论 |
| XGBoost 验证 | `/api/run_xgb_validation`、XGB 结果文件 | 仅显示 XGB 验证真实结果；不回写初步评分或排序 | `未执行` / 无对应结果文件 |

## 结论

本任务不是整体回滚。应撤销初步分析对闭环风险、候选类别、驱动优先级和四层解释的产品消费，恢复 `final_score` 稳定降序作为初步筛选、Top-K、推荐输出和 Web 默认排序依据；同时保留缺失值语义、权重重归一、滞后方向、数据质量、代理识别及确定性合成基准等统计正确性修复。后续分析算法和路由保持独立，只有真实执行并生成对应结果文件后才展示其结果。
