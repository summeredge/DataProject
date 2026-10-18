# 第一阶段分析质量优化验收记录（F1 / F2 / F4）

日期：2026-10-18。三个独立修改单元按 PR-1 → PR-2 → PR-3 顺序实施；这里的 PR 名称仅表示修改单元，没有创建 GitHub PR、提交或推送。

## 修改单元

| 单元 | 实际修改 | 独立审查 |
| --- | --- | --- |
| PR-1 / F1 | 预处理保留各变量的时间点与缺失值；按有限插值后的工况掩码，在插值前数据上计算所选工况有效率；保持全时段历史来源；差分、去趋势及创新量不再联合删除任意列缺失行。同步 service / pipeline / web 调用。 | Hypatia 首轮发现插值前工况校验 P2；修复并补充反例后复审通过，无未解决 P0/P1/P2。 |
| PR-2 / F2 | 最佳峰和峰值质量共用每变量有效样本门槛；不改原始 lag_scores 统计值、V5 公式、Top-K 或排序契约。 | Russell 审查通过，无确认 P0/P1/P2；独立复跑 115 项通过。 |
| PR-3 / F4 | 合成评估读取正式输出的四类 CSV，调用实际二级输入选择函数；分别报告召回、重要变量缺失集合、适用规则、能力边界与候选传递成功状态。 | Fermat 审查通过，无确认 P0/P1/P2；独立运行完整合成测试文件 57 项通过。 |

## PR-2 规则选择证据

每个变量在当前搜索滞后集合中的最大有效配对数为 `max_n`，选峰要求：

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
| 有界 50%（本次采用） | 0/10 | 2 | 170 |

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
