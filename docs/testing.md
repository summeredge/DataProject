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

## PR-TR2 趋势页排除窗口测试边界

- 现有趋势鼠标选区直接用于加入窗口，不新增第二套选择状态；
- 单窗口、多窗口和重叠窗口的列表、统计与趋势背景标记一致，统计复用
  `exclude_window_stats()`；
- 单窗口恢复只移除指定窗口，恢复全部数据只清空 `exclude_windows`，不改写上传数据；
- 新上传数据上下文不继承旧窗口；排除窗口操作不调用或改变初筛及后续分析流程。

## 数据质量风险测试

数据质量测试覆盖以下行为：

-   普通质量超限（如 `missing_rate = 0.21`）只写入 `poor_data_quality`，
    仅提示并平滑降分：不封顶、不升级强风险计数、不强制
    `candidate_class = poor_quality`、不强制
    `recommended_use = poor_quality_variable`；
-   严重质量超限（`missing_rate > 0.50`、`saturation_ratio > 0.80`、
    `abnormal_jump_ratio > 0.05`、`robust_outlier_ratio > 0.05`）只写入
    `severe_data_quality`，作为一次强风险计数，`final_score` 上限为
    `0.44`，`risk_cap_reason = severe_data_quality`；
-   普通与严重标记互斥，严重条件不得同时出现 `poor_data_quality`；
-   严重阈值使用严格大于，恰好等于阈值不触发严重风险；
-   `data_quality_score` 连续衰减且四项为零时为 `1.0`；
-   `ranked_features.csv` 仍按 `final_score` 降序，CSV/API 字段结构不变。

历史兼容测试覆盖：`poor_data_quality` 与旧 `poor_quality` 分类、旧推荐用途或旧
`medium` 风险等级并存时，仍只作为普通质量提示；精确
`severe_data_quality` 才触发综合复核硬降级和 XGBoost 自动排除。排名基线的质量风险
及 A/B 风险计数按唯一变量断言，重复行不得重复计数。

## Raw 主筛查回归基线测试

使用项目内可控的小型固定数据（不依赖随机外部数据），锁定 Raw 主筛查行为：

- `final_score` 数值；
- `final_score` 降序排序；
- `driver_rank`；
- 初筛 Top-K；
- 候选池变量及来源；
- `best_lag` 的正负方向（`ranked_features.csv` 的 `lag` 列）；
- `ranked_features.csv` 关键字段；
- 新配置字段采用默认值时 Raw 输出不变。

浮点断言使用合理容差，但不得宽泛到掩盖评分变化。

## 预处理模式与配置字段测试

- `lowpass_tau_minutes <= 0` 被明确拒绝；
- `diff_interval_minutes <= 0` 被明确拒绝，`None` 合法；
- 四种契约模式（`raw`、`lowpass`、`lowpass_detrend`、`lowpass_diff`）可被配置对象
  表示；
- `transform_frame()` 支持 `lowpass*`；
- `transform_frame_causal()` 支持 `lowpass*`；
- causal 新模式不得使用未来样本；
- `analyze_numeric_frame()` 当前仍拒绝 `lowpass*`；
- `run_initial_screening_workflow()` 接受 `raw` / `lowpass*`，旧模式明确
  拒绝；Web/CLI 正式入口只暴露 `raw` / `lowpass*` 四种模式；
- Raw 和旧模式保持回归不变；
- `transform_frame()` 的基础能力不得自动进入评分、排序、候选池；
- 旧模式（`detrend`、`diff`、`detrend_diff`）保持兼容，且不被映射为新模式；
- 新增字段和模式不得进入评分、排序、候选池或默认执行流程。

## 单分支初筛 Runner 测试边界

- `run_initial_screening_branch()` 一次调用只执行一个分支，不自动执行另一分支；
- `branch=raw` 只允许 `preprocess_mode=raw`；
- `branch=processed` 只允许 `lowpass` / `lowpass_detrend` / `lowpass_diff`；
- 非法 branch 或 branch/mode 不匹配必须明确 `ValueError`，不得静默纠正；
- `raw` 分支输出隔离到 `screening_branches/raw/`，`processed` 分支输出隔离到
  `screening_branches/processed/`；
- 分支 runner 不得向运行根目录发布正式初筛文件；
- Raw 分支结果与正式 `analyze_numeric_frame()` / `run_analysis()` 的 Raw 结果
  一致；
- `lowpass_tau_minutes` 与 `diff_interval_minutes` 必须真实传入
  `transform_frame()`；
- 正式 `analyze_numeric_frame()` / `run_analysis()` 仍拒绝三个 `lowpass*` 模式；
- 两个分支输出互不覆盖，同分支重跑只覆盖自身分支；
- 不生成 `preprocessing_comparison.csv`、`preprocessing_context.json`；
- 不实现 confirmation / promotion。

## 双分支 comparison runner 测试边界

- `run_initial_screening_comparison()` 只允许 `lowpass` / `lowpass_detrend` /
  `lowpass_diff`，`raw` 与旧模式必须明确拒绝；
- 一次调用分别独立执行 raw 与 selected processed 两个分支，且不改写调用方
  `AnalysisConfig`；
- 两个分支仍输出到 `screening_branches/raw/` 与 `screening_branches/processed/`，
  不创建按 processed mode 命名的第三层目录；
- 双分支均成功后生成且只生成一个 `preprocessing_comparison.csv`（utf-8-sig），
  字段与顺序完全符合冻结契约；
- `variable` 为两个 `ranked_features` 的有序并集（raw 顺序 + 仅 processed 变量按
  processed 顺序）；
- `final_score_delta = processed - raw`，`rank_delta = raw_rank - processed_rank`；
- lag 保持正负方向，`lag_direction_changed` 按 negative/zero/positive 区分，
  任一侧缺失时保持缺失；
- Top-K 与 candidate membership 分别按各分支独立计算；
- risk 直接来自各分支 `risk_flags`，变量不存在侧保持缺失，不得用空字符串冒充
  无风险；comparison CSV 中缺失侧为 `NaN`，变量存在但无风险时为空字符串；
- comparison 不修改任何 branch 结果、不合并候选池、不产生推荐分支；
- 任一分支失败时不得生成半成品 comparison；
- 不生成 `preprocessing_context.json`，不向运行根目录发布正式初筛结果；
- 不实现 confirmation / promotion / Web / API / CLI 交互。

## 统一 workflow 与 branch confirmation 测试边界

- `run_initial_screening_workflow()` 只允许 `raw` / `lowpass` /
  `lowpass_detrend` / `lowpass_diff`，旧模式必须明确 `ValueError` 且不得清理
  已有 context / root / comparison；
- 已存在 `screening_downstream.lock` 时再次启动 workflow 必须拒绝
  （`initial_screening_run_locked`），现有运行文件不得改变；
- `raw` workflow 只运行 raw branch，不运行 processed，不生成
  `preprocessing_comparison.csv`；raw 自动 promotion 到正式 root，context
  状态为 `not_required`，active branch/mode 均为 `raw`；
- 非 Raw workflow 复用 `run_initial_screening_comparison()` 运行 raw +
  processed 双分支并生成 comparison，context 状态为
  `awaiting_confirmation`，active branch/mode 在 JSON 中必须真实为 `null`，
  正式 root 初筛文件不得存在；
- `preprocessing_context.json` 字段集合固定为冻结契约，不新增评分、排序或
  自动推荐字段；
- `lowpass*` 的 `lowpass_tau_minutes` 记录用户配置；差分字段为 `null`；
- `lowpass_diff` 的 requested/effective diff 参数复用 `resolve_diff_interval()`
  与现有采样周期 / resample 规则：1 min 数据、`diff_interval_minutes=5` 得
  requested `5.0` / points `5` / interval `5.0`；`None` 得 requested `null` /
  points `1` / interval `1.0`；
- `resample_rule` 记录真实配置；
- `confirm_initial_screening_branch()` 只允许 `raw` / `processed`，必须读取
  context；确认 raw 后 `active_preprocessing_mode = raw`，确认 processed 后
  `active_preprocessing_mode` 为 selected `lowpass*`，
  `selected_preprocessing_mode` 不得改写；
- confirmation 不得调用 `run_initial_screening_branch()` /
  `run_initial_screening_comparison()` / `analyze_initial_screening_branch_frame()`
  （确认 ≠ 重新运行）；
- 同 branch 重复确认必须幂等（成功返回、状态不变、不重新分析、不创建重复
  文件）；
- downstream 开始前允许 `confirm raw → confirm processed` 切换，root 最终
  必须精确等于新 branch；
- branch 缺少 9 个必需正式文件时 confirmation 必须失败
  （`initial_screening_branch_output_incomplete`），context 保持原状态，root
  不得出现半成品；
- promotion 是文件级发布：staging → backup → replace → context 更新，任一
  replace / context 写入失败必须回滚原 root 文件与原 context，不允许
  Raw/Processed 混合 root；
- 新 branch 缺少 `residual_corr_scores.csv` 时，旧 root 的可选文件残留必须被
  删除且进入 rollback 事务；
- `begin_downstream_stage()` 读取 context：缺失
  `initial_screening_context_missing`、非法 `initial_screening_context_invalid`、
  `awaiting_confirmation` 时 `initial_screening_branch_not_confirmed`（不得创建
  lock）；`confirmed` / `not_required` 通过并创建 `screening_downstream.lock`；
- lock 后切换到另一 branch 必须拒绝（`initial_screening_branch_locked`），
  root / context 不得改变；lock 后同 branch 确认允许 no-op；
- 未锁定目录重新用于非 Raw workflow 时，上一轮正式 root 文件与旧 context
  必须失效，新 comparison 完成后不得继续暴露上一轮 `ranked_features.csv` /
  `recommended_candidates.csv`。

## PR-13 Web / API / CLI 正式接入测试边界

`tests/test_pr13_web_preprocessing_workflow.py` 与
`tests/test_pr13_cli_workflow.py` 覆盖正式双分支工作流总接入：

- Web 与 CLI 的预处理模式选择只包含 `raw` / `lowpass` / `lowpass_detrend` /
  `lowpass_diff`，旧模式不再出现在正式选择中；
- `lowpass_tau_minutes` 默认 `5.0`，随表单/CLI 参数传入 `AnalysisConfig`；
  `diff_interval_minutes` 空值为 `None`，正数正常传递，非正数明确拒绝；
- `/api/analyze` 与 CLI `analyze` 调用 `run_initial_screening_workflow()`，
  正式入口不再调用 `run_analysis()`；
- Raw 只运行 Raw 并自动 promotion，返回正式 payload，不生成 comparison；
- 非 Raw 双分支完成后返回 pending payload：
  `branchSelectionStatus = awaiting_confirmation`、
  `activeScreeningBranch = null`、`activePreprocessingMode = null`、
  `preprocessingComparison` 非空，正式 root 初筛文件不存在，且不把分支内部
  文件包装为正式 `rankedFeatures`；
- comparison 数据直接来自冻结 `preprocessing_comparison.csv`，不重新计算
  delta / rank；
- 确认 API（`POST /api/confirm_initial_screening_branch`）与 CLI
  `confirm-branch` 只调用 `confirm_initial_screening_branch()`，执行期间
  `run_initial_screening_workflow` / `run_initial_screening_branch` /
  `run_initial_screening_comparison` 调用次数为 0；
- 确认 raw / processed 后 payload 分别返回
  `activeScreeningBranch = raw/processed`、
  `activePreprocessingMode = raw/selected lowpass*`，且
  `selectedPreprocessingMode` 保持最初比较模式；
- downstream 开始前允许 raw → processed 切换；存在
  `screening_downstream.lock` 后切换必须拒绝
  `initial_screening_branch_locked`；
- `awaiting_confirmation` 时增强筛选 / Granger / Model / 三级复核 /
  XGBoost 五个 Web endpoint 全部明确阻断（`initial_screening_branch_not_confirmed`），
  不得自动 Raw fallback；
- 五个 downstream Web endpoint 与 CLI `run-*` 命令只调用对应
  `run_*_for_active_branch()` 正式 runner；Web endpoint 源码不得包含
  `run_analysis(` / `run_xgb_analysis(` / `run_causal_review_stage(` /
  `fit_explainable_model(` / `run_granger_tests(` /
  `_build_causal_review_candidate_table(` /
  `_load_secondary_candidate_context(` 用于正式执行；
- XGB Web API 走 `run_xgb_for_active_branch()` fold-safe 路径，不再使用
  `_prepared_frame_for_validation() + run_xgb_analysis()`；
- 三级复核运行前后 `causal_review_candidates.csv` byte-identical，
  `risk_flag_filter` 只过滤展示结果；
- UI 静态契约：`awaiting_confirmation` 时 downstream 按钮禁用并提示
  “请先确认正式初筛分支。”；`confirmed/not_required` 时启用；
  `screening_downstream.lock` 存在时另一分支确认按钮禁用并提示锁定文案；
- payload 场景 `selected = lowpass_diff`、`confirmed = raw` 必须同时返回
  `selectedPreprocessingMode = lowpass_diff` 与
  `activePreprocessingMode = raw`（`analysisContext.preprocess_mode = raw`）；
- `preprocessing_comparison.csv` 与 `preprocessing_context.json` 可合法下载，
  `screening_branches/raw/*`、`screening_branches/processed/*` 与 `../*`
  仍被拒绝；
- LLM Prompt / 综合报告在 `awaiting_confirmation` 时拒绝
  `initial_screening_branch_not_confirmed`；
- CLI `run-*` 命令以 `--output RUN_DIR` 读取已有
  `run_config.json` / `preprocessing_context.json` / formal root，不要求重传
  初筛参数；processed + `--enable-granger/--enable-model` 不得自动选择
  branch，必须提示先 `confirm-branch`；Raw + enable 标志在 promotion 后调用
  对应正式 runner；
- 测试不得实际训练 XGB。
- 图表回归覆盖 selected processed / confirmed raw 使用 Raw，confirmed
  processed 使用正式 tau/diff/detrend 参数，分析后修改表单不覆盖 active
  context，pending 仍是预览且不产生 active branch；后端继续调用回顾性
  `transform_frame()` 并明确拒绝非有限或非正 tau/diff。

## PR-9 增强筛选 / 普通 Granger 正式 branch/context 测试边界

`tests/test_pr9_active_branch_downstream.py` 覆盖正式 downstream 入口
（`run_enhanced_screening_for_active_branch()` /
`run_granger_for_active_branch()`）：

- `awaiting_confirmation` 时两个入口都必须拒绝
  `initial_screening_branch_not_confirmed`，不生成阶段结果、不创建
  `screening_downstream.lock`；
- `selected_preprocessing_mode = lowpass_diff` 但确认 Raw 时，两个入口实际
  收到 `preprocess_mode = raw`，不得使用 selected 模式；
- 确认 Processed（`lowpass_diff`）时，`preprocess_mode`、
  `lowpass_tau_minutes`、`diff_interval_minutes`、`resample_rule` 必须与
  context 一致；
- 四个 PR-9～PR-11 正式 runner 调用统一 causal secondary frame helper，
  不调用 legacy 回顾性 helper；future suffix 变化不得改变既有 prefix，
  predictor 缺失不得使用未来值做双向插值；
- Enhanced 强制禁止复用 ranked lag evidence；Model 对全部实际候选重算
  causal signed lag；三级复核只接收内存 causal ranked lag view，初筛 CSV
  仍 byte-identical；
- Raw workflow（`not_required`）下两个入口可直接运行并使用 `raw`；
- context 是 source of truth：调用方传入冲突的 `preprocess_mode` /
  `lowpass_tau_minutes` / `diff_interval_minutes` 不得覆盖
  `active_preprocessing_mode` 与 context 参数；
- 候选只来自正式 root：非 active branch（如 `screening_branches/processed/`）
  中的 sentinel 变量不得进入后续分析；
- 增强筛选 / Granger 运行前后 `ranked_features.csv`、
  `recommended_candidates.csv`、`causal_review_candidates.csv` 必须
  byte-identical，`final_score` / `driver_rank` / 候选顺序不得改变；
- 第一个 downstream stage 创建 `screening_downstream.lock`；增强筛选创建
  lock 后普通 Granger 仍可继续运行，不得报
  `initial_screening_run_locked`；
- context 已确认但正式 root 缺少 `recommended_candidates.csv` 等必要输入时，
  两个入口都必须失败 `initial_screening_formal_output_missing`，不创建
  lock，不得读取 branch 目录补救；
- context 缺失 / JSON 非法 / 状态非法分别失败
  `initial_screening_context_missing` / `initial_screening_context_invalid`，
  不得 fallback；
- 普通 Granger 保留 signed lag：正式 runner 源码不得使用
  `abs(lag)` / `abs(best_lag)` / `abs(granger_lag)`，输出
  `best_granger_lag` 保持真实正滞后；
- 只执行增强筛选 / Granger 时不得自动生成 conditional Granger、causal
  review、RF/SHAP/model discovery、XGBoost 结果文件。

## PR-10 RF / SHAP / Model Discovery 正式 branch/context 测试边界

`tests/test_pr10_active_branch_model.py` 覆盖正式模型 backend 入口
（`run_model_for_active_branch()`）：

- `awaiting_confirmation` 时拒绝 `initial_screening_branch_not_confirmed`，
  不生成模型输出、不创建 `screening_downstream.lock`；
- `selected_preprocessing_mode = lowpass_diff` 但确认 Raw 时模型实际收到
  `preprocess_mode = raw`，不得使用 selected 模式；
- 确认 Processed 时 `preprocess_mode` / `lowpass_tau_minutes` /
  `diff_interval_minutes` / `resample_rule` 与 context 一致；
- Raw workflow（`not_required`）下可直接运行并使用 `raw`；
- context 是 source of truth：调用方冲突的 preprocessing config 不得覆盖
  active 模式与 context 参数；
- 模型候选只来自正式 root `ranked_features.csv`，非 active branch 的
  sentinel 变量不得传入 `fit_explainable_model()`；
- 模型候选沿用 Top-K + force include 去重规则，不得重新出现
  `final_score >= 0.30` 截断；
- 模型复用现有 `fit_explainable_model()` / `build_model_variable_importance()`
  / `build_model_discovered_candidates()`，参数（target、max_lag、
  candidate_variables、max_features、random_state、best_lags、
  `lag_mode = best_only`、target_mask）来自正式 downstream config；
- 成功运行只生成 `shap_or_importance.csv`、`model_variable_importance.csv`、
  `model_discovered_candidates.csv`，不新增综合评分 CSV；
- 运行前后 `ranked_features.csv` / `recommended_candidates.csv` /
  `causal_review_candidates.csv` byte-identical，`final_score` /
  `driver_rank` / 变量顺序不变；
- model discovery 不回写初筛；风险信息来自正式 root `risk_flags.csv`，
  不使用非 active branch 风险；
- 模型作为第一个 downstream stage 时创建 lock；已有 lock（增强筛选 /
  Granger 先运行）后模型仍可运行；模型运行后切换 branch 必须拒绝
  `initial_screening_branch_locked`；
- 正式 root 缺少 `ranked_features.csv` / `risk_flags.csv` 时失败
  `initial_screening_formal_output_missing`，不创建 lock、不读 branch 目录；
- context 缺失 / JSON 非法 / 状态非法分别失败
  `initial_screening_context_missing` / `initial_screening_context_invalid`；
- SHAP 不可用时仍可运行并产出 `random_forest_feature_importance`，不得让
  正式 runner 失败；
- 只执行模型时不得自动生成 conditional Granger、causal/final review、
  XGBoost 等未执行阶段文件；
- 正式 runner 源码不得使用 `abs(lag)` / `abs(best_lag)`，不得读取
  `screening_branches/` 构造候选。

## PR-11 Conditional Granger / Causal Review / Final Review 正式 branch/context 测试边界

`tests/test_pr11_active_branch_causal_review.py` 覆盖正式三级复核 backend
入口（`run_causal_review_for_active_branch()`）：

- `awaiting_confirmation` 时拒绝 `initial_screening_branch_not_confirmed`，
  不生成四个三级输出、不创建 `screening_downstream.lock`；
- `selected_preprocessing_mode = lowpass_diff` 但确认 Raw 时三级实际收到
  `preprocess_mode = raw`，不得使用 selected 模式；
- 确认 Processed 时 `preprocess_mode` / `lowpass_tau_minutes` /
  `diff_interval_minutes` / `resample_rule` 与 context 一致；
- context 是 source of truth：调用方冲突的 preprocessing config 不得覆盖
  active 模式与 context 参数；
- 正式三级候选唯一来自 root `causal_review_candidates.csv`，不得从
  ranked / secondary candidate context / 非 active branch /
  model discovered candidates 重新生成或扩大；执行前后
  `causal_review_candidates.csv` byte-identical；
- 正式三级复核显式启用 causal ranked-lag precedence：候选表历史 lag 为 20、
  内存 causal lag 为 2 时 ranked-window 为 `[1, 2, 3]`；causal lag 缺失时沿用
  `fallback_missing_ranked_lag`，不得回退历史 lag。standalone 默认仍保持
  候选表 lag 优先；
- `top_n=None` 默认覆盖完整正式候选集，包含 `final_score < 0.30` 但仍属
  正式候选的变量；显式 `top_n` 保留原顺序传给 stage，由 stage 现有
  `head(top_n)` 截断，不得重新排序；
- control columns 解析优先级：显式 → `residual_control_columns` →
  `capacity_columns` → `[]`；显式空列表不回退 config；显式控制列通过
  causal secondary frame helper 的 `protected_columns=...` 保护；
- `target` / `control_columns` / `maxlag` / `min_rows` / `top_n` /
  conditional 参数 / `target_mask` 全部完整透传现有
  `run_causal_review_stage()`；`maxlag=None` 使用现有
  `config.resolved_granger_maxlag()`；
- signed lag 保持方向：`ranked_features.csv` 中负 lag 原样传入 stage，
  正式 runner 源码不得使用 `abs(lag)` / `abs(best_lag)` /
  `abs(granger_lag)`；
- 运行前后 `ranked_features.csv` / `recommended_candidates.csv` /
  `causal_review_candidates.csv` / `risk_flags.csv` byte-identical，
  `final_score` / `driver_rank` / 候选顺序不变；
- 成功执行只生成四个三级输出
  （`conditional_granger_scores.csv`、`causal_review_report.csv`、
  `causal_review_evidence.csv`、`final_review_summary.csv`），不新增综合
  score/rank 文件，不生成其他阶段文件；
- optional evidence 缺失时三级仍可执行，不自动运行对应前置阶段；已有
  optional evidence（`enhanced_validation_summary.csv`、
  `model_variable_importance.csv` 等）可被现有 stage 从 `output_dir`
  读取，`pipeline.py` 不复制 optional evidence 读取逻辑；
- 三级作为第一个 downstream stage 时创建 lock；已有 lock（增强筛选 /
  Granger / Model 先运行）后三级仍可运行；三级运行后切换 branch 必须拒绝
  `initial_screening_branch_locked`；
- 正式 root 缺少四个必需输入中任一文件时失败
  `initial_screening_formal_output_missing`，不创建 lock、不生成三级输出、
  不读 branch 目录；context 缺失 / JSON 非法 / 状态非法分别失败
  `initial_screening_context_missing` / `initial_screening_context_invalid`；
- 只执行三级复核时不得自动生成 Enhanced / ordinary Granger / Model /
  XGBoost 等未执行阶段文件；
- 正式 runner 源码约束：不得包含 `_save_secondary_candidate_context` /
  `_load_secondary_candidate_context` / `_build_causal_review_candidate_table`
  / `build_causal_review_candidates` / `screening_branches` /
  `preprocessing_comparison.csv` / `model_discovered_candidates` /
  `final_score >= 0.30` / `abs(lag` / `abs(best_lag`。

## PR-12 XGBoost 正式 branch/context 与 fold preprocessing isolation 测试边界

`tests/test_pr12_active_branch_xgb.py` 覆盖正式 XGB backend 入口
（`run_xgb_for_active_branch()`）：

- `awaiting_confirmation` 拒绝 `initial_screening_branch_not_confirmed`，
  不生成 `xgb_validation/`、不创建 `screening_downstream.lock`；
- 缺失 `final_review_summary.csv` 明确失败
  `initial_screening_formal_output_missing`，不自动运行三级复核；
- 确认 Raw 优先于 `selected_preprocessing_mode`；确认 Processed 时
  `preprocess_mode` / `lowpass_tau_minutes` / `diff_interval_minutes` /
  `resample_rule` 与 context 一致；caller config 不得覆盖 context；
- control columns / whitelist / top_n / max_lag 解析优先级保持现有契约；
- 只消费正式 root `ranked_features.csv` 与 `final_review_summary.csv`，
  不读 `screening_branches/`、`preprocessing_comparison.csv` 或
  `model_discovered_candidates.csv`；
- 运行前后前三层正式文件 byte-identical；成功只写五个 XGB 输出文件；
- 首次成功创建 downstream lock，已有 lock 仍可继续运行；
- 缺失 xgboost 返回 `missing_dependency`；
- 正式入口返回有效样本不足导致的 `invalid_input`（例如工况 mask 使
  effective train rows < 100）；正式入口 `row_count == len(xgb_predictions.csv)`。

`tests/test_xgb_fold_preprocessing_isolation.py` 覆盖 fold 状态隔离：

- gap 等于实际 max used lag，train/validation/test 无时间重叠；
- 改变 train 尾部值不得改变 validation/test 的独立 transform 结果
  （lowpass / lowpass_detrend / lowpass_diff）；
- diff 首值不跨 partition 借用上一分区；forward-fill 不跨 partition；
- partition 内真实物理缺口仍按现有 lowpass / diff / ffill 规则重启；
- raw 分支仍使用 expanding split、gap 与 positive lag；
- 每个 fold 在 preprocessing / target mask / lag / dropna 后再次校验
  100 / 30 / 30 有效样本下限，不足时返回 `invalid_input` 且不训练任何模型，
  已有五个输出保持不变；
- lowpass_diff 或 lag 删除导致有效样本不足时正确阻断；
- `row_count` 等于实际 out-of-time prediction rows，不等于 split-base 行数；
- `data_fingerprint` 覆盖所有 fold 实际 train / validation / test 输入：修改
  任意 test / validation / 非首 fold 实际数据都会改变 fingerprint，相同输入
  重复执行稳定，未进入 feature 的无关列变化不影响 fingerprint。
