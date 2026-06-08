# 化工装置工业时序数据筛查

本项目用于工业装置历史时序数据的四层筛查。用户指定一个目标变量后，程序会从大量过程变量中快速筛出相关性线索，并结合残差相关、工况稳定性、风险标签和预测提升给出候选变量排序。

默认策略强调效率：第一阶段默认不运行 PCMCI、Transfer Entropy、XGBoost、LightGBM、全量 DTW，也默认不运行 SHAP。Granger 与模型解释保留为二阶段复核接口，不参与主评分。

## 第一阶段默认分析方法（已更新）

默认只执行轻量方法：

- 数据清洗：时间排序、重复时间戳处理、数值列选择、缺失插值
- 可选重采样：例如 1min、5min、15min
- 标准化：消除量纲影响
- Pearson 相关：线性同步和滞后关系
- Spearman 相关：单调非线性关系，比 Pearson 稳健
- p 值、q 值与 r²：输出每个滞后点的统计显著性与多重比较校正（Pearson/Spearman）
- 工况分段：可按负荷代表列切分低/中/高负荷或自定义范围
- 去趋势/差分：减少共同趋势造成的伪相关
- 滞后扫描：在 `[-max_lag, +max_lag]` 范围内寻找最强相关滞后
- 残差相关：支持多列残差控制（residual control columns），对 target 与 candidate 分别多元回归残差化后再做滞后相关
- 工况稳定性：输出低/中/高工况下的强度、符号一致性、滞后一致性及稳定性指标
- 变量排序：综合 raw/residual 相关、工况稳定性、滚动稳定性、lag 峰值质量、model lift、风险惩罚输出候选变量
- 方向判断：同步变化、变量领先目标、变量滞后目标

## 后续可选方法

只建议在初筛后，对 Top 变量小范围启用：

- Granger：轻量时序预测因果候选，使用 `--enable-granger`
- 随机森林/SHAP：变量解释，使用 `--enable-model`

暂不建议默认使用：

- PCMCI / PCMCI+
- Transfer Entropy
- XGBoost / LightGBM
- 全量 DTW

这些方法更适合在变量数量已经大幅缩小后使用，或者有明确业务需求时单独追加。

## 快速开始

安装 Python 3.10 或以上版本后，在项目目录安装基础依赖：

```powershell
pip install -e .
```

## 推荐使用方式：本地 Python 服务 + 浏览器界面

双击项目根目录的：

```text
start_app.bat
```

它会自动：

- 检查 Python
- 检查并安装基础依赖
- 启动本地 Python 服务
- 打开浏览器访问 `http://127.0.0.1:8765/`

使用流程：

1. 浏览器上传 CSV、TSV、TXT 或 Excel（`.xlsx`、`.xls`、`.xlsm`）数据。
2. 选择编码、时间列、目标列。
3. 设置滞后、预处理、工况分段等参数。
4. 点击“开始分析”。
5. Python 后台处理数据。
6. 浏览器展示候选变量排序，并提供结果文件下载。

结果会保存到：

```text
reports/web_runs/
```

如果需要更精确的 p 值、Granger 或模型解释，安装完整依赖：

```powershell
pip install -e ".[full]"
```

小数据也可用无需后台服务的静态界面：

直接用浏览器打开 `ui/index.html`。这个版本完全在浏览器内运行，适合本地快速初筛。数据较大时建议使用下面的 Python 命令行版本。

如果希望通过本地服务访问，也可以启动本地交互界面：

```powershell
python -m chem_ts_corr.cli serve
```

打开浏览器中的本地地址后，可以通过界面完成：

- 上传 CSV、TSV、TXT 或 Excel 数据
- 选择时间列
- 选择目标变量
- 设置最大滞后点数、Top K、有效数据比例、重采样规则，并可在大数据场景跳过模型提升评分/滚动稳定性评分以加速
- 点击“开始分析”
- 查看可调整宽度的结果表格，并下载候选变量排序、滞后明细等 CSV 结果

运行轻量初筛：

```powershell
python -m chem_ts_corr.cli analyze `
  --input examples/sample_plant_timeseries.csv `
  --time-column timestamp `
  --target reactor_temp `
  --max-lag 12 `
  --top-k 30 `
  --output reports/demo
```

大数据建议使用 Python 命令行：

```powershell
python -m chem_ts_corr.cli analyze `
  --input D:\data\plant_history.csv `
  --encoding gb18030 `
  --time-column 时间 `
  --target 目标变量 `
  --max-lag 24 `
  --top-k 50 `
  --preprocess-mode detrend `
  --detrend-window 48 `
  --segment-column 负荷 `
  --segment-mode mid `
  --residual-control-columns 负荷,进料量 `
  --force-include-variables 关键变量A,关键变量B `
  --output reports/big_data_run
```

常用参数：

- `--input`：支持 CSV、TSV、TXT 与 Excel 文件；TXT/TSV 会自动尝试分隔符，Excel 依赖 `openpyxl` / `xlrd`
- `--encoding`：文本文件编码，中文 Windows CSV/TXT 常用 `gb18030`
- `--preprocess-mode`：`raw`、`detrend`、`diff`、`detrend_diff`
- `--detrend-window`：滑动均值去趋势窗口点数
- `--segment-column`：负荷代表列（仅用于工况分段）
- `--segment-mode`：`all`、`low`、`mid`、`high`、`custom`
- `--segment-min / --segment-max`：自定义工况范围
- `--residual-control-columns`：残差相关控制列（多列逗号分隔）
- `--capacity-columns`：向后兼容别名（等价 residual control columns）
- `--force-include-variables`：强制进入滚动稳定性复核的变量列表
- Web 大数据加速选项：可勾选“跳过模型提升评分（大数据加速）”和“跳过滚动稳定性评分（大数据加速）”，用于缩短主筛查耗时；这两个选项不会改变 `ranked_features.final_score` 的计算公式，只会将对应模块输出标记为跳过/缺省。
- Web 结果表格：多数结果表支持横向滚动，右下角可拖动调整表格宽度，便于查看宽表字段。

如果后续要对筛选后的变量追加 Granger：

```powershell
python -m chem_ts_corr.cli analyze `
  --input examples/sample_plant_timeseries.csv `
  --time-column timestamp `
  --target reactor_temp `
  --max-lag 12 `
  --top-k 30 `
  --enable-granger `
  --output reports/demo_granger
```

如果要追加模型解释，需要安装完整依赖：

```powershell
pip install -e ".[full]"
```

然后运行：

```powershell
python -m chem_ts_corr.cli analyze `
  --input examples/sample_plant_timeseries.csv `
  --time-column timestamp `
  --target reactor_temp `
  --max-lag 12 `
  --top-k 30 `
  --enable-model `
  --output reports/demo_model
```

## 输出文件

- `summary.md`：分析摘要
- `ranked_features.csv`：候选变量排序主表
- `recommended_candidates.csv`：推荐候选（含 `candidate_grade`、`recommended_use`）
- `lag_scores.csv`：各变量/各滞后点明细，包含 `effective_n`、`pearson_q`、`spearman_q`、`corr_q_value`、`lag_boundary_flag`
- `lag_peak_quality.csv`：lag 峰值质量（`best_lag`、`peak_sharpness`、`lag_quality`）
- `diagnostics.csv`：缺失、长缺失段、异常跳变比例、饱和比例等数据质量诊断
- `residual_corr_scores.csv`：剔除一个或多个 CAPACITY 控制列后的残差相关
- `regime_scores.csv`：低/中/高工况相关结果与稳定性指标（含符号一致性、滞后一致性、CV 等）
- `rolling_corr_scores.csv`：滚动相关稳定性（`rolling_corr_median`、`rolling_corr_iqr`、`rolling_sign_consistency`、`rolling_stability`）
- `risk_flags.csv`：数据质量、公式型变量、共同驱动、闭环、目标领先、跨工况不稳定、时序不稳定、lag 边界、低提升等风险标记
- `model_lift_scores.csv`：TimeSeriesSplit 下 AR baseline 与 AR + candidate lag features 的误差改善
- `granger_tests.csv`：默认跳过，启用 Granger 后输出结果
- `shap_or_importance.csv`：默认跳过，启用模型解释后输出特征级重要性明细
- `model_variable_importance.csv`：模型解释变量排序，按变量汇总随机森林/SHAP 重要性且每个变量仅展示最强 lag；该文件不代表因果结论，不改变主筛查综合得分。
- `model_discovered_candidates.csv`：模型解释补充候选，用于发现主筛查可能遗漏的非线性/多滞后预测线索；该文件不代表因果结论，不改变主筛查综合得分。
- `near_miss_candidates.csv`：轻量遗漏候选，基于已有滞后相关、残差相关、峰值质量和风险标签提示主筛查 Top K 外可能遗漏的候选；该文件不代表因果结论，不改变主筛查综合得分。

## 推荐工作流

1. 在界面上传 CSV。
2. 选择时间列和目标变量。
3. 设置最大滞后窗口。
4. 点击“开始分析”完成轻量初筛。
5. 查看 Top 变量及其领先/滞后方向。
6. 剔除明显由控制回路、共同趋势、开停车过程造成的伪相关。
7. 只对剩余候选变量追加 Granger、PCMCI 或人工机理复核。

## 滞后方向解释

- `lag > 0`：变量领先目标，可能是目标变化的前置信号。
- `lag = 0`：变量与目标同步变化。
- `lag < 0`：变量滞后目标，更可能是结果变量、反馈响应或共同扰动后的响应。

## 注意事项

相关性初筛不能证明工艺因果。工业装置里常见的伪相关来源包括共同负荷变化、控制回路、物料停留时间、批次切换、牌号切换、开停车、仪表漂移和数据压缩策略。

因此，初筛结果应作为“候选线索”，不是最终结论。
