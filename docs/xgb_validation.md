# XGB 四级验证说明

## 1. 目的

XGB 四级验证用于检查候选变量能否在严格时间外测试中，为基线模型提供非线性增量预测价值。它是第四级预测验证，不是因果识别，也不参与前三层正式评分与排序。

## 2. 前置条件

必须先完成三层复核并生成 `final_review_summary.csv`。Web 默认不启用 XGB；只有用户显式勾选并运行后才会训练模型和写入 `xgb_validation/`。

运行环境需要可选依赖：

```powershell
pip install -e ".[xgb]"
```

## 3. 模型定义

- `M0`：目标变量自身的正滞后历史。
- `M1`：`M0` 加控制变量的正滞后历史。
- `M2`：`M1` 加全部候选变量的有效正滞后特征。
- `Candidate_i`：`M1` 加单个候选变量 `i` 的有效正滞后特征。

整体模型比较使用 M0/M1/M2；逐候选 uplift 使用 `Candidate_i` 与同一时间折的 M1 比较。M1 结果从整体模型验证复用，不重复训练。

## 4. 数据口径

XGB 继承前三层分析配置中的：

- IGNORE 角色过滤；
- 工况分段；
- 重采样；
- 缺失插值与完整样本处理；
- `raw`、`detrend`、`diff`、`detrend_diff` 变换。

XGB 不执行标准化。`screening_lag` 的点数单位因此与重采样后的数据间隔一致。所有模型使用同一份完整样本，输入 DataFrame、`ranked_features` 和 `final_review_summary` 不会被原地修改。

## 5. 时间切分

验证采用 expanding time split，不使用随机切分。每个时间折按时间顺序分为 train、validation 和 test，并在相邻分区间保留与最大使用滞后相同的 gap：

- train 用于拟合；
- validation 仅用于 early stopping；
- test 仅用于最终 RMSE、MAE 和 R2 指标，不参与模型选择。

所有特征只使用正滞后，不读取未来值。

## 6. 状态解释

- `validated_incremental_signal`：多个时间折稳定改善，中位 RMSE 改善为正且 MAE 不退化。
- `weak_incremental_value`：中位 RMSE 有改善，但稳定性或折数尚不足以视为稳定增量信号。
- `redundant_with_baseline`：相对 M1 没有正向中位改善，候选信息可能已被基线覆盖。
- `unstable_out_of_time`：不同未来时间折同时出现改善和退化，时间外表现不稳定。
- `insufficient_features`：候选没有可用正滞后特征，保留记录但不训练候选模型。

## 7. 输出文件

输出位于运行目录的 `xgb_validation/`：

- `xgb_fold_metrics.csv`：M0/M1/M2 各时间折指标。
- `xgb_model_summary.csv`：整体模型跨折摘要及 M2 相对 M1 的改善。
- `xgb_candidate_uplift.csv`：逐候选增量验证摘要和状态。
- `xgb_predictions.csv`：各测试折真实值与 M0/M1/M2 预测。
- `xgb_validation_summary.json`：数据规模、特征规模、配置、provenance fingerprint、阶段耗时和文件清单。

JSON 不包含原始数据值、用户文件路径或前三层排名字段。

## 8. 性能说明

默认自动候选数量为 8，用户可将自动候选数量调整到最多 10。白名单候选可在自动候选之外追加；自动候选与白名单合并、规范化去重并排除目标变量后，总候选数量最多为 12。超过 12 会返回 `invalid_input`，不会静默截断或运行部分候选。

设最终实际候选数为 `C`、时间折数为 `F`，完整流程训练次数为：

```text
(3 + C) x F
```

前三项对应 M0、M1、M2；候选模型为 `C x F`。`C` 为最终实际候选数量，包含白名单候选，最大为 12。候选数量与运行时间近似线性增长。XGB 训练可能占用较多 CPU，建议先使用默认 TopN 8 和默认模型参数。`max_lag` 必须在 1～5000 之间；过大会减少有效完整样本并扩大时间 gap，不应盲目增加。

可重复性能基准：

```powershell
python scripts/benchmark_xgb_validation.py --rows 50000 --variables 50 --candidates 8 --max-lag 360
```

## 9. 故障排查

- `missing_dependency`：安装 `pip install -e ".[xgb]"` 后重试。
- `invalid_input`：检查目标列、TopN、`max_lag`、时间索引和必要输入表。
- `failed`：查看错误信息，确认内存、CPU 和 XGBoost 运行环境可用。
- 样本不足：减少滞后、调整重采样或扩大时间范围；不要使用随机切分规避。
- 滞后过大：将 `max_lag` 调回与工艺停留时间和采样间隔一致的范围。
- 没有有效候选：检查正滞后、候选列是否存在，以及白名单变量是否有可用特征。
- `final_review_summary` 缺失：先运行三层复核并确认 `final_review_summary.csv` 已生成。

## 10. 解释边界

预测增量不等于因果成立，不代表变量可操纵，也不代表变量适合进入 APC。XGB 不会消除闭环反馈、共线性、共同负荷、公式泄漏或数据质量风险。结果只用于独立人工解释，不会修改 `final_score`、`driver_rank`、`final_rank`、候选等级或风险标签，也不会自动删除候选。
