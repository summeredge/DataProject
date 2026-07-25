# 四层证据归属、重复计分和排序契约审计（PR-8A）

范围：`main` 的筛选排序链路。此审计不改变任何公式、权重、阈值、风险规则或输出。`ranked_features.csv` 的正式排序入口是 `screening.final_ranked_features`，排序键是 `driver_rank`（由 `driver_priority_score` 降序生成）。`causal_review_evidence` 和 `final_review_summary` 是筛选后的独立人工复核链路；它们会形成复核优先级，但不会回写筛选分数或 `ranked_features.csv`。

## 计算契约

`correlation_evidence_score` 由 association、innovation、independence 中实际可用项几何组合。`evidence_strength` 是多个允许权重 profile 在实际可用评分组件上重新归一化后所得分数的中位数；`evidence_confidence = data_quality_score`；`evidence_score = evidence_strength × evidence_confidence`；`final_score` 为 evidence_score 经明确风险扣减和风险上限处理后的结果；`driver_priority_score = final_score × driver_priority_factor`，并由其降序产生 `driver_rank`。未计算、不可用或数据不足的可选评分组件从 profile 中省略并重新归一化，不按零分处理；实际计算得到 `0.0` 才属于弱证据。`evidence_completeness` 与 `evidence_coverage_status` 仅作评分组件覆盖输出，不进入 evidence_score 或排序。

当前 profile 的四个组成项权重均为 0.10--0.40，合计为 1；并非单一固定权重。类别 factor：upstream 1.00、synchronous 0.90、downstream 0.45、capacity 0.75、formula 0.25、poor_quality 0.35、uncertain 0.80。风险相对扣减仅为 `strong_formula_leakage=0.50`、`residual_collinearity=0.10`；风险上限仅为 `strong_formula_leakage=0.25`、`poor_data_quality=0.44`。等级阈值：A/B/C/D 为 `final_score >= 0.75/0.60/0.45/0.30`，其余 E。`top_k` 按 `driver_rank` 截取，强制包含变量追加但不重排其全局 rank；控制变量从候选截取中排除并覆盖 `recommended_use` 为参考用途。

## 字段注册表

机器可读版本见 `docs/four_layer_evidence_registry.json`。下表以一行一个实际证据或输出字段记录来源、消费者和排序语义；“直接”指进入 `final_score` 或 `driver_priority_score` 的数值表达式，“间接”指经合成、风险、类别或输出政策进入。

|字段名|生成文件 / 函数|使用文件 / 函数|层级|证据|直接评分|间接评分|重复/重叠|缺失值|负面结果|当前权重或 factor|后续处理|本 PR修改|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|raw_corr|lag.py / summarize_best_lags|screening.py / final_ranked_features|L1|原始|否|association_score|与 innovation、residual 同属关联证据|填 0|绝对关联非负|无|保留，后续评估慢趋势重复|否|
|association_score|screening.py / final_ranked_features|screening.py / _combine_correlation_evidence|L1|派生|否|是|association_combined|raw_corr 缺失为 0|截断 [0,1]|几何组合|审查 raw/innovation 语义边界|否|
|innovation_score|service.py / _innovation_evidence|screening.py / final_ranked_features|L1|派生|否|是|与 association、residual 几何合成|not_computed，降 coverage|0 是已计算负证据|几何组合 + coverage|验证慢趋势校正是否应双重影响|否|
|residual_corr|screening.py / residual_corr_scores|screening.py / final_ranked_features,risk_flags|L3|派生|否|是|与原关联同源、但控制共同负荷|not_computed，省略|截断 [0,1]|几何组合|评估与 common_capacity 风险的重复约束|否|
|independent_signal_score|screening.py / final_ranked_features|screening.py / _combine_correlation_evidence|L3|派生|否|是|residual_corr 的标准化别名|not_computed，省略|截断 [0,1]|几何组合|无公式改动；后续拆分独立性语义|否|
|correlation_evidence_score|screening.py / _combine_correlation_evidence|screening.py / final_ranked_features|L1/L3|派生|是|否|汇总 association/innovation/independence|无 association 则 NaN|几何均值可为 0|profile 组成项|可能把关联证据多次表达，需实验审查|否|
|lag|lag.py / summarize_best_lags|screening.py / classify_candidate,risk_flags,_recommend_use|L2|原始|否|是|与 lag_quality、Granger 都表达时间性|缺失为 uncertain|负 lag 为 target leads|类别/风险，无数值权重|对照 Granger，避免未来重复奖励|否|
|lag_quality|lag.py / build_lag_peak_quality|screening.py / final_ranked_features|L2|派生|是|否|与 lag、rolling/regime 时间稳定性重叠|not_computed，降 coverage|截断 [0,1]|profile 0.10--0.40|评估其与 Granger 的语义重叠|否|
|lag_boundary_flag|lag.py / build_lag_peak_quality|screening.py / risk_flags|risk|派生|否|风险提示|与 lag_quality 边界信息重叠|缺失为 false|仅 flag|当前 0 penalty|保留为提示，不等同无效|否|
|rolling_stability|screening.py / rolling_corr_scores|screening.py / final_ranked_features,risk_flags|stability|派生|否|是|与 regime_stability 合并|not_computed，可用 regime|<.35 flag|profile 经 stability_score|检查直接评分与风险/等级的间接重叠|否|
|regime_stability_final|screening.py / _summarize_regime_robustness|screening.py / final_ranked_features,risk_flags|stability|派生|否|是|与 rolling_stability 合并|not_computed，可用 rolling|<.50 flag（已评估时）|profile 经 stability_score|同上|否|
|stability_score|screening.py / final_ranked_features|screening.py / profile score|stability|派生|是|否|rolling/regime 的几何或 fallback 聚合|not_computed，降 coverage|非负几何均值|profile 0.10--0.40|后续判断风险与等级联动|否|
|model_lift_score|screening.py / model_lift_scores|screening.py / final_ranked_features,risk_flags,_recommend_use|L4|派生|否|是|与 causal-review model_lift、importance、conditional contribution 不同链路|not_computed，省略|低 lift 可 flag|profile 经 prediction_score|检查复核层多模型证据是否重复奖励|否|
|prediction_score|screening.py / final_ranked_features|screening.py / profile score|L4|派生|是|否|model_lift_score 的 gated 别名|非 ok 状态为 missing|0 为已计算无增益|profile 0.10--0.40|无公式改动；后续统一模型语义|否|
|data_quality_score|screening.py / _data_quality_score|screening.py / final_ranked_features|data_quality|派生|是|否|与 poor_data_quality cap 相关|风险表缺失默认 1|低质量也触发 cap|confidence 因子|检查平滑扣减与 cap 的双重约束|否|
|evidence_completeness|screening.py / final_ranked_features|CSV/API/report|评分组件覆盖|派生|否|否|四个评分组件及关联族变化量证据的覆盖信息|缺失组件不按零分|不适用|不参与评分|保持缺失和失败分离|否|
|evidence_confidence|screening.py / final_ranked_features|screening.py / evidence_score|data_quality|派生|是|否|data_quality_score 的兼容别名|派生|不适用|乘 evidence_strength|后续只做审计|否|
|evidence_coverage_status|screening.py / final_ranked_features|CSV/API/report|评分组件覆盖|派生|否|否|评分组件是否完整|缺失组件不按零分|不适用|不参与评分|展示覆盖，不参与排序|否|
|evidence_missing_items|screening.py / final_ranked_features|CSV/API/report|评分组件覆盖|派生|否|否|缺失的变化量、模型、稳定性或滞后质量评分组件|缺失组件清单|不适用|不参与评分|展示覆盖，不参与排序|否|
|four_layer_missing_items|evidence_explanations.py / add_evidence_explanations|CSV/API/report/web|四层解释覆盖|派生|否|否|Layer 1～4、稳定性和数据质量解释中未获得或数据不足的项|只列 not_available/insufficient_data|不适用|不参与评分|与评分组件覆盖分开解释|是|
|four_layer_coverage_status|evidence_explanations.py / add_evidence_explanations|CSV/API/report/web|四层解释覆盖|派生|否|否|六个解释状态是否均已获得|完整/部分完整/证据不足|不适用|不参与评分|完整不代表全部支持|是|
|evidence_strength/evidence_score|screening.py / final_ranked_features|screening.py / 风险调整|聚合|派生|是|否|四个 profile 组件|无可用项为 NaN|0 为已计算弱证据|profile 中位数、confidence|审查组件重叠，不变更|否|
|risk_flags/risk_level|screening.py / risk_flags|screening.py / _risk_adjustment,classify_candidate,_recommend_use|risk|派生|否|是|风险同时影响 penalty/cap/class/use|空集合|flag 可能只提示|见上文 penalty/cap|风险不可直接解释为变量无效|否|
|risk_penalty_rate/risk_score_cap|screening.py / _risk_adjustment|screening.py / final_ranked_features|risk|派生|是|否|risk_flags 的数值投影|无 flag 为 0/1|扣减或封顶|0.50/0.10；0.25/0.44|审查 data quality 的 factor+cap 重叠|否|
|final_score|screening.py / final_ranked_features|screening.py / _finalize_driver_ranking,_grade_candidate|聚合|派生|是|否|风险调整后的 evidence_score|派生|已扣风险|最终统计分|冻结基线|否|
|candidate_class|screening.py / classify_candidate|screening.py / factor|工程优先级|派生|否|是|lag/risk 共同决定|未知 lag 为 uncertain|保守类别|0.25--1.00 factor 选择器|与风险的再约束需后续评审|否|
|driver_priority_factor/score/rank|screening.py / final_ranked_features,_finalize_driver_ranking|screening.py / 排序；web.py / overview|工程优先级|派生|score/rank 是|是|final_score 加类别 factor|派生|不适用|class factor；降序 rank|冻结为主排序契约|否|
|candidate_grade|screening.py / _grade_candidate|report.py, causal_review_evidence.py|输出|派生|否|后续复核间接使用|final_score fallback 0|D/E|0.75/0.60/0.45/0.30|不能与复核 score 混称|否|
|recommended_use/action|screening.py / _recommend_use,_recommended_action|report.py, causal_review_evidence.py|输出|派生|否|政策/复核间接使用|default manual review|风险限制用途|无|不作为排序键|否|
|conditional_fdr_q_value / predictive_contribution|causal_review.py / conditional Granger|causal_review_evidence.py / _assess_row|L3/L4|派生|否（筛选）|仅复核排序|无结果为 insufficient|无支持不等于失败|0.05/0.10 review 阈值|与筛选 Granger 完全隔离|否|
|granger_fdr_q_value|causality.py / run_granger_tests|causal_review_evidence.py / _assess_row|L2|派生|否（筛选）|仅复核排序|无辅助支持|高 q 无支持|0.05/0.10 review 阈值|与 lag evidence 的重复奖励仅存在复核层|否|
|model_importance_rank / model_lift|modeling.py / fit_explainable_model；screening.py / model_lift_scores|causal_review_evidence.py / _assess_row|L4|派生|筛选仅 model_lift|复核再计分|missing 无支持|低排名/无 lift 无支持|复核加分|XGB/SHAP 不回写筛选排序|否|
|engineering_context|外部工程元数据|CSV/API/report|工程上下文|派生|否|否|与四层统计证据隔离|空字符串|不适用|无|仅作上下文展示，不参与排序|否|

## 重复计分与历史残留结论

## 逐字段更正登记（替代上表所有斜杠合并写法）

以下每行均为独立字段；来源、消费者、缺失与负面语义的机器可读完整值以 registry 对应对象为准。本表补齐主排序 `model_lift` 回退、复核评分和最终复核排序字段。

|字段名|生成函数|使用函数|层级|直接评分/排序|缺失值与负面结果|本 PR修改|
|---|---|---|---|---|---|---|
|model_lift|model_lift_scores|final_ranked_features,_assess_row|L4|主排序回退输入；复核直接加分|缺失时不作支持；低值负证据|否|
|predictive_contribution|run_conditional_granger_tests|_assess_row|L4|复核直接加分|缺失/非正不加分|否|
|conditional_fdr_q_value|run_conditional_granger_tests|_assess_row|L3|复核直接加分|缺失为证据不足；高 q 无支持|否|
|granger_fdr_q_value|run_granger_tests|_assess_row|L2|复核直接加分|缺失/高 q 无支持|否|
|model_importance_rank|fit_explainable_model|_assess_row|L4|复核直接加分|缺失/排名靠后无支持|否|
|rolling_sign_consistency|rolling_corr_scores|_assess_row|stability|复核直接加分|缺失/低值无支持|否|
|evidence_score|_assess_row|_evidence_level,final_review_summary|复核聚合|复核排序键|无支持时较低；非筛选分|否|
|evidence_level|_evidence_level|_integrated_decision,final_review_summary|复核聚合|复核决策输入|缺失转 not_supported|否|
|data_priority|_data_priority|final_review_summary|复核聚合|复核排序键|缺失转 low|否|
|integrated_review_decision|_integrated_decision|build_final_review_summary|复核聚合|生成最终复核排序键|缺失转 not_recommended|否|
|final_recommendation|build_final_review_summary|build_final_review_summary sort|复核聚合|最终复核第一排序键|缺失转 not_recommended|否|
|final_rank|build_final_review_summary|CSV/report|复核聚合|最终复核输出 rank|由复核 sort 产生|否|
|screening_score|build_final_review_summary|build_final_review_summary sort|复核聚合|最终复核第四排序键|缺失为 -inf|否|
|screening_lag|build_final_review_summary|final_review_summary|L2|复核输出|缺失为空|否|
|risk_constraint_level|_risk_constraint_level|final_review_summary|risk|复核限制|缺失为 none|否|
|statistical_limit_level|_statistical_limit_assessment|_integrated_decision,final_review_summary|risk|复核限制|缺失为 none|否|
|engineering_context|外部工程元数据|CSV/API/report|工程上下文|不参与评分/排序|空字符串|否|

工程上下文与四层统计证据、推荐用途和排序字段隔离；它只作为单独输出字段展示，不参与评分、等级或排序。

1. 筛选总分中，原始关联不会以独立加法项重复进入；但 `association_score`、`innovation_score`、`independent_signal_score` 被几何合并，属于同一关联族的多证据约束。是否构成“重复奖励”需要后续用对照数据评估，当前不能据此改公式。
2. lag 相关/lag_quality 进入筛选；Granger 只在筛选后因果复核中加分。因此不存在 Granger 回写 `final_score` 的重复奖励，但复核层确实同时使用 lag、Granger 和条件 Granger，必须与筛选排序分开解释。
3. 独立性在筛选层只经 `residual_corr -> independent_signal_score` 进入总分；`common_capacity_driver` 是风险分类，当前无相对 penalty，但可改变类别 factor/用途。存在语义重叠，不是两次直接数值加分。
4. 筛选层仅使用 `model_lift_score`；XGBoost、SHAP/importance、conditional contribution 不回写筛选。它们可在复核层与 model lift 并列加分，属于复核优先级的潜在模型证据重叠。
5. stability 直接通过 `stability_score` 入 profile，并通过不稳定 flag 改变用途；当前不稳定 flag 的相对 penalty 为 0，且不直接限制 candidate grade，所以不是“直接加分又等级封顶”。
6. data quality 既进入 `evidence_confidence`，又可触发 `poor_data_quality` 的 0.44 cap 和类别/用途限制，确有双重约束；本 PR 只记录并冻结。
7. optional 证据缺失会从当前 profile 省略并重新归一化，不会被替换为零分；已计算的 0 则保留为负面证据。`evidence_completeness`、`evidence_coverage_status` 与 `evidence_missing_items` 仅描述评分组件覆盖；四层解释覆盖由 `four_layer_*` 字段单独输出。风险 flag 是约束/提示，除列明的 penalty/cap 外不等价于变量无效。
8. 工程上下文仅通过 `engineering_context` 输出；它不读取进分数、factor、等级、推荐用途或排序。

## 入口覆盖与冻结

测试 `tests/test_four_layer_evidence_audit.py` 覆盖直接函数入口 `final_ranked_features`、服务入口 `analyze_numeric_frame`、管线入口 `run_analysis` 写出的 `ranked_features.csv`，并冻结 `final_score`、`driver_priority_factor`、`driver_priority_score`、`driver_rank`、`candidate_grade`、`recommended_use` 以及 CSV 关键字段和顺序。它还检查 registry 的直接评分字段能在评分函数中找到、所有已知排序输入都已登记、工程上下文字段没有进入评分表达式。现有 `test_causal_review_evidence.py` 和 `test_final_review_summary.py` 覆盖两条后续复核排序实现；其排序不属于 `ranked_features.csv` 契约。
