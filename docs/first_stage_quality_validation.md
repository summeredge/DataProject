# 第一阶段分析质量优化验收记录（F1 / F2 / F4）

日期：2026-10-18。三个独立修改单元按 PR-1 → PR-2 → PR-3 顺序实施；这里的 PR 名称仅表示修改单元，没有创建 GitHub PR、提交或推送。

## 首轮修改单元（历史验收记录）

| 单元 | 实际修改 | 独立审查 |
| --- | --- | --- |
| PR-1 / F1 | 预处理保留各变量的时间点与缺失值；按有限插值后的工况掩码，在插值前数据上计算所选工况有效率；保持全时段历史来源；差分、去趋势及创新量不再联合删除任意列缺失行。同步 service / pipeline / web 调用。 | Hypatia 首轮发现插值前工况校验 P2；修复并补充反例后复审通过，无未解决 P0/P1/P2。 |
| PR-2 / F2 | 最佳峰和峰值质量共用每变量有效样本门槛；不改原始 lag_scores 统计值、V5 公式、Top-K 或排序契约。 | Russell 审查通过，无确认 P0/P1/P2；独立复跑 115 项通过。 |
| PR-3 / F4 | 合成评估读取正式输出的四类 CSV，调用实际二级输入选择函数；分别报告召回、重要变量缺失集合、适用规则、能力边界与候选传递成功状态。 | Fermat 审查通过，无确认 P0/P1/P2；独立运行完整合成测试文件 57 项通过。 |

## PR-2 初版规则选择证据（历史）

以下记录首轮规则；现行规则的下限已在本次补充修复中改为 8，详见文末补充验收。初版每变量最大有效配对数为 `max_n`，选峰要求：

`n >= clip(ceil(0.5 * max_n), 5, 20)`

5 保留既有底线；20 的上限避免仅因重叠比例较低而丢掉有充分绝对样本的长滞后。原始相关系数、样本数和 NaN 不改写；选峰、近峰区间、峰值质量与方向判断共享过滤结果。没有可计算峰值时保持空结果，不填真实零值。

固定种子 0～9、40 行独立噪声、搜索 ±35 点，比较结果：

| 规则 | 噪声获 A/B 的次数 | 10 行已知 lag=2 | 240 行已知 lag=170 |
| --- | --- | --- | --- |
| 原规则（至少 5 对） | 7/10 | 原有短序列行为保留为比较基准 | 原有长滞后行为保留为比较基准 |
| 固定至少 10 对 | 2/10 | 误选 0 | 170 |
| 固定至少 15 对 | 0/10 | 无可计算峰 | 170 |
| 固定至少 20 对 | 0/10 | 无可计算峰 | 170 |
| 单纯最大配对数的 50% | 0/10 | 2 | 误选 111 |
| 有界 50%（PR-2 初版） | 0/10 | 2 | 170 |

另有正负 lag=170 的回归，70 对有效样本均正确保留。门槛只是一条经过本组合成反例验证的选峰规则，不代表独立样本量估计或统计关系保证。

本机抽样：500 行、3 个候选、25 个 lag；100 次重复摘要与峰值质量计算，每次从 6.57 ms 增至 8.38 ms。相关搜索仍调用原算法和相同次数；单次搜索测得约 720 ms（含首次导入成本），不是严格性能基准。

## 候选召回证据

`tests/baselines/initial_screening_baseline.json` 每个场景增加 `candidate_coverage`：

- `sets`：`ranked_features.csv`、`recommended_candidates.csv`、`verification_review_pool.csv`、`actual_validation_variables`、`causal_review_candidates.csv` 的实际有序变量。
- `recall`：各集合对已知重要变量的覆盖率。没有真实驱动的噪声案例为 `null`，不伪造零召回或成功。
- `important_variables`：逐变量成员状态、缺失集合、首次缺失集合、适用的既有规则、初筛名次、分数、等级与风险。
- `candidate_transfer_success` 与原有契约 `passed` 分开；已知算法能力不足不会被排名存在掩盖。

| 场景 | 完整排名召回 | 一级推荐召回 | 二级复核池 / 实际验证输入召回 | 第三层候选召回 |
| --- | --- | --- | --- | --- |
| 已知正向滞后 | 1 | 1 | 1 / 1 | 1 |
| 共线冗余变量中的已知驱动 | 1 | 1 | 1 / 1 | 1 |
| 所选工况完整、其他时段缺失 | 1 | 1 | 1 / 1 | 1 |
| 非单调平方关系 | 1 | 0 | 1 / 1 | 0 |
| 工况符号反转 | 1 | 0 | 1 / 1 | 0 |
| 40 行低样本噪声 | 无真实驱动，召回不适用 | 不据噪声入池宣称召回成功 | 保留既有候选规则 | 最高等级不再为 A/B |

非单调及符号反转案例在一级推荐处缺失；二级按完整排名 Top-K 仍可能纳入，第三层则沿用一级推荐派生集合。这是当前 F3 契约下的预期集合差异，本次没有修改候选传递。另以混合证据场景 Top-K=1 断言实际二级输入遗漏也能单独报告，并覆盖人工指定变量的来源与顺序。

生成候选集合复用 `write_outputs()`、`write_initial_verification_review_pool()` 和 `_verification_review_variables()`；未执行后续 Enhanced、Granger、模型解释或第三层完整分析。

基线扩展对 HEAD 的原有 11 个场景逐字段比较（数值容差 1e-12），所有旧字段、旧分数、旧排名和旧验收要求均未改变。新增 2 个场景及配对数 n、候选覆盖记录。

## 验证记录

解释器固定为 `C:\Users\shaoy\AppData\Local\Programs\Python\Python311\python.exe`（3.11.9）；NumPy 2.3.5、pandas 2.3.3、SciPy 1.16.3、pytest 9.0.3。未安装依赖或切换到 Codex bundled runtime。

以下均使用该解释器执行 `-m pytest -q -p no:cacheprovider`（早期验证未禁用 cache 时有本机缓存目录权限警告）：

- PR-1：缺失变量、时间轴、排除窗口及旧预处理保护测试最新相关集 63 项通过；包含 18 个新增回归。另一次分支 runner、downstream、secondary lag、物理差分和低通测试 128 项通过。此前完整 Raw 基线、初筛契约及新变换相关用例亦通过；不同批次不累加为一次总数。
- PR-2：样本支持、lag、峰值质量、方向、时间轴、V5、F1 和 Raw 初筛契约相关集 185 项通过。追加残差、二级 lag、时序关联、排名和 Web 相关证据检查 116 项通过、1 项已存在失败。
- PR-3：合成与复核池检查 60 项通过（当时排除基线匹配）；最终新增候选行为集 8 项通过；完整基线匹配及探索池、发现候选契约集 9 项通过。以上有重叠，不相加。
- 最终 F1 / F2 / F4 集成检查：42 项通过，覆盖缺失工况候选、低样本噪声、阶段集合差异、人工加入来源、Top-K 遗漏、初筛隔离与复核池契约。

最终集成命令（在仓库根目录执行，PowerShell 续行）：

```powershell
$python = "C:\Users\shaoy\AppData\Local\Programs\Python\Python311\python.exe"
& $python -m pytest -q -p no:cacheprovider `
  tests/test_initial_screening_contract.py `
  tests/test_verification_review_pool.py `
  tests/test_initial_screening_synthetic_baseline.py::test_complete_rank_is_not_candidate_transfer_success `
  tests/test_initial_screening_synthetic_baseline.py::test_important_driver_reaches_actual_candidate_sets `
  tests/test_initial_screening_synthetic_baseline.py::test_coverage_reads_formal_outputs_and_actual_validation_selector `
  tests/test_initial_screening_synthetic_baseline.py::test_low_sample_noise_has_no_invented_driver_recall `
  tests/test_initial_screening_synthetic_baseline.py::test_top_k_omission_is_visible_in_actual_second_stage_input
```

结果：`42 passed in 8.09s`。最终 `git diff --check` 通过。

复杂度 Audit（no-stop-gpt）：工况参数均有 service / pipeline / web 实际消费者；共享样本门槛由选峰及质量两处复用；删除了保留时间轴后不可达的删行 warnings 分支；候选评估直接复用正式 writer 和选择函数，没有新增生产框架、依赖、缓存或兼容层。基线增加的是各集合实际覆盖证据。

## 既有问题与限制

- `tests/test_web_correlation_evidence.py::test_analysis_context_is_replaced_and_cleared_without_reading_current_form` 在当前代码及 HEAD 的 `INDEX_HTML` 上均复现同一断言失败；本次没有改动页面上传逻辑。
- HEAD 合成基线已记录三项未满足能力期望：`common_driver` 的共同驱动抑制、`outlier_driven_correlation` 的离群高等级、`lag_boundary_artifact` 的边界等级上限。本次如实保留全部原始失败记录，没有改评分/质量惩罚来消除它们。
- 未验证真实工业数据、浏览器 UI 或昂贵的完整后续分析。本次证据来自可重复合成数据、函数/API/文件契约和独立代码审查。
- F3、F5、评分权重、初筛 Top-K 配置、后续阶段隔离、依赖版本均未修改。

## 修改文件

- PR-1：`chem_ts_corr/preprocess.py`、`service.py`、`pipeline.py`、`web.py`；`tests/test_missing_candidate_samples.py`、`test_time_axis_lag_semantics.py`、`test_v05_pr_a_correctness.py`。
- PR-2：`chem_ts_corr/lag.py`；`tests/test_lag_sample_support.py`、`test_temporal_near_peak_direction.py`。
- PR-3：`tests/synthetic_cases/evaluate.py`、`four_layer_cases.py`、`generate_baseline.py`；`tests/test_initial_screening_synthetic_baseline.py`、`tests/baselines/initial_screening_baseline.json`。
- 本验收记录：`docs/first_stage_quality_validation.md`。


## F1 / F2 补充修复与最终验收（现行版本）

本次基于已完成的三个修改单元补充修复，不重做 PR-1～PR-3。现行选峰门槛为 `n >= clip(ceil(0.5 * max_n), 8, 20)`；V5 评分、初筛排序、Top-K、二级候选集合和既有 PR-3 评估机制均未修改。

### F1：工况前历史来源

- 目标工况选择规则不变。复用 `lagged_series()`，把目标掩码按允许正 lag 反向映射为合法来源掩码；遵守物理采样周期、排除窗口及已有 forced breaks，不把目标时刻裁成连续序列。
- 保持目标时刻的既有有效率过滤，同时保留合法历史来源上存在测量的候选。准入只保守保留待评估列；实际有效配对数和可否选峰仍由统一 lag 计算负责，不再用扩大搜索后被稀释的历史全窗口有效率提前删列，也不新增逐候选的第二次滞后相关扫描。
- 仅第一阶段在目标掩码内完全无可用测量的列，使用允许 lag 范围内的历史值补齐标准化均值和尺度。有目标时刻拟合数据的列保持原拟合结果；更早但不可配对的数据不参与该补齐。后续 causal / XGB 标准化流程不改。
- 重采样后立即更新既有 `lag_sample_period_ns` 属性，避免来源掩码读取旧采样周期。该问题由本轮独立 Review 发现并补回归后修复。
- 合成回归覆盖：仅切换前有驱动数据，切换后选中 40 个目标时刻；lag=20 有 20 对、lag=40 有 40 对；搜索范围分别为真实 lag 的 1 倍和 2 倍，均保留候选且正确识别 lag。标准化不整列 NaN，目标拟合不变。超出范围、真实排除窗口和物理断点均不能贡献历史准入与拟合。
- 重采样回归比较“无周期属性”和“保存原始 1 分钟周期”两种输入；重采样为 2 分钟后得到相同结果。
- 另复用原 PR-3 `run_case()` / 正式 CSV / 实际验证输入选择器，验证历史驱动进入四类候选集合及实际二级输入，无需新增评估框架或运行昂贵后续模型。

### F2：窄工况和短有效窗口

30 个独立噪声反例：40 行数据，分别选首、中、末各 10 个目标时刻，±35 搜索；每种位置种子 0～9。单纯提高支持比例至 60%～80%不能可靠解决低样本峰，故保留 50%比例及 20 对上限，仅调整原有绝对下限。

| 最少配对下限 | 选中不足 8 对且获 A/B 的反例数 | 全部 A/B 数（包括 8～10 对） | 10 行真实 lag=2 |
| --- | --- | --- | --- |
| 5 | 6/30 | 21/30 | lag=2，n=8 |
| 6 | 2/30 | 19/30 | lag=2，n=8 |
| 7 | 1/30 | 20/30 | lag=2，n=8 |
| 8（采用） | 0/30 | 18/30 | lag=2，n=8 |
| 9 | 0/30 | 17/30 | 误选 lag=-1，n=9 |

8 是本组合成反例中排除 5～7 对虚高峰、同时保留既有 10 行真实滞后峰的局部选择。正负 lag=2 均保留；240 行真实 lag=±170 的 70 对峰，以及原 40 行噪声用例继续通过。最佳峰、近峰区间和方向分类继续共用同一有效集合，没有新增评分或独立统计算法。

固定 10 个目标时刻、候选只有 5/6/7 点有效的回归，保留原始相关与 n，但不产生充分支持的最佳峰或质量/方向结果；不填零相关。既有真实零值回归继续通过。

这里没有宣称消除所有窄工况噪声高等级：8～10 对样本仍可能出现偶然高相关。该统计风险如实保留；本次没有修改评分公式以额外压低这些等级。计算流程只改既有门槛常量，F2 不增加搜索、相关计算或新的复杂度。

### 补充独立 Review 与验证

- F1：Anscombe 首轮发现重采样周期 P2；修复后独立复审 PASS，无未解决 P0/P1/P2，独立运行缺失样本、时间轴与排除窗口共 60 项通过。主 Agent 另验证相关初筛、Raw 和 causal 契约；分批测试有重叠，不求和。
- F2：Carson 独立 Review PASS，无确认 P0/P1/P2；独立运行相关测试 112 项通过，最新短有效窗口 fixture 另验 3 项通过（不累加）。主 Agent 已运行 lag 样本支持、方向、峰值质量、最佳峰证据、检测和统计对照测试共 70 项通过；加强后的 6 个窄工况断言及 3 个短有效窗口断言另行通过。
- F1/F2/F4 最终集成：`22 passed in 33.46s`。覆盖仅历史有数据的驱动进入现有候选集合、带原周期属性的重采样、窄工况、10 行真实正负滞后、短有效窗口不可充分评估、13 场景完整基线匹配、一级推荐与二级输入差异、Top-K 遗漏和复核池契约。
- PR-3 原评估、生成器、既有合成测试和基线 JSON 与本轮开始时保持一致（`git diff --quiet --` 对相关文件返回 0），没有为通过验收重写基线。

最终集成命令：

```powershell
$python = "C:\Users\shaoy\AppData\Local\Programs\Python\Python311\python.exe"
& $python -m pytest -q -p no:cacheprovider `
  tests/test_missing_candidate_samples.py::test_historical_only_driver_is_visible_to_existing_candidate_coverage `
  tests/test_missing_candidate_samples.py::test_resampled_history_uses_new_period_even_with_original_period_metadata `
  tests/test_lag_sample_support.py::test_narrow_regime_cannot_select_five_to_seven_pair_noise_peaks `
  tests/test_lag_sample_support.py::test_ten_row_true_relationship_keeps_eight_pair_peak `
  tests/test_lag_sample_support.py::test_computable_but_unsupported_short_window_has_no_peak `
  tests/test_initial_screening_synthetic_baseline.py::test_committed_initial_baseline_matches_actual_report `
  tests/test_initial_screening_synthetic_baseline.py::test_complete_rank_is_not_candidate_transfer_success `
  tests/test_initial_screening_synthetic_baseline.py::test_top_k_omission_is_visible_in_actual_second_stage_input `
  tests/test_verification_review_pool.py
```

最终 `git diff --check` 通过。

解释器与首轮相同，所有本次 pytest 命令均使用系统 Python 3.11 的绝对路径与 `-p no:cacheprovider`。未安装依赖。

复杂度 Audit（no-stop-gpt）：来源掩码由准入和受限标准化两处复用，只操作布尔时间掩码；不逐列重跑相关计算，不新增缓存/框架。历史补齐有明确缺失均值触发条件，保留第一阶段消费者与 downstream 拟合边界。F2 仅复用原共享门槛。

本次文件归属：

- `chem_ts_corr/preprocess.py`：来源掩码、保守历史准入、受限标准化、重采样周期同步。
- `chem_ts_corr/service.py`：第一阶段传递 `max_lag` 给预处理和标准化。
- `chem_ts_corr/pipeline.py`、`chem_ts_corr/web.py`：已有预处理调用传递 `max_lag`；web 本任务只涉及这一行。
- `tests/test_missing_candidate_samples.py`、`tests/test_lag_sample_support.py`：补充边界与接口回归。
- 本文档：补充修复、实验、Review 和验收结果。

工作期间还出现非本任务修改，归属未确认：`web.py` 下拉面板 UI，数值排除相关的 `config.py`、`data.py`、`docs/contracts.md`，`pipeline.py` / `web.py` 的其他 hunks，以及 `tests/test_web_variable_select_search.py`、`tests/test_web_history_reanalysis.py`、`tests/test_exclude_conditions.py`。均保留，未纳入本次实现或审查通过结论。

本次 F1、F2 补充修复与必要 F1/F2/F4 集成验收通过。没有执行真实工业数据、浏览器 UI 或完整后续模型验收；8～10 对噪声高等级的风险仍如上所述。不创建 PR，不提交、不推送，保留工作区修改，等待用户手动推送。
