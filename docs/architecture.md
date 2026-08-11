# Architecture

## 分析流程

数据输入 → 数据预处理 → 第一阶段：主筛查 → 候选池 → 第二阶段：增强验证 →
第三阶段：综合复核 → 第四阶段：时间外预测验证

## 阶段边界

主筛查负责生成候选。

后续分析只能提供补充证据，不得修改：

-   final_score
-   初筛排序
-   Top-K

## 数据流

允许：

主筛查 → 候选池 → 后续验证

禁止：

-   模型结果 → 初筛评分
-   综合复核 → 覆盖初筛结果
-   未执行分析 → 展示分析结论

## 预处理模式

正式 `analyze_numeric_frame()` / `run_analysis()` 当前实际执行模式：

```text
raw
detrend
diff
detrend_diff
```

预处理基础能力：

```text
transform_frame():
  支持 lowpass / lowpass_detrend / lowpass_diff

transform_frame_causal():
  支持 lowpass / lowpass_detrend / lowpass_diff
```

正式 `analyze_numeric_frame()` / `run_analysis()` 仍拒绝：

```text
lowpass
lowpass_detrend
lowpass_diff
```

模式语义与配置字段约束见 `docs/contracts.md`。`lowpass*` 模式可在配置对象中表示，
但正式分析入口执行前必须被明确拒绝，不得静默回退到 `raw`。

单 branch 初筛 runner 已实现：

```text
run_initial_screening_branch():
  branch=raw       → preprocess_mode 必须为 raw
  branch=processed → preprocess_mode 只能为 lowpass / lowpass_detrend / lowpass_diff
```

一次调用只执行一个分支，结果写入：

```text
run_directory/
└─ screening_branches/
   ├─ raw/
   └─ processed/
```

## 初筛双分支

第一阶段分支产物目标目录：

```text
run_directory/
├─ screening_branches/
│  ├─ raw/
│  └─ processed/
├─ preprocessing_comparison.csv
└─ preprocessing_context.json
```

隔离约束：

- 分支候选池不得合并；
- 不得按变量取两分支较高分数；
- 不得生成新的综合评分；
- 未确认前不得在运行根目录发布正式初筛文件；
- 确认后只发布选定分支的结果。

当前已实现：

- 单 branch 独立运行（`run_initial_screening_branch()`）；
- branch 输出隔离到 `screening_branches/raw/` 或 `screening_branches/processed/`；
- branch runner 不向运行根目录发布正式初筛文件。

当前尚未实现：

- Raw + Processed 自动双分支 orchestration；
- `preprocessing_comparison.csv`；
- `preprocessing_context.json`；
- branch confirmation；
- promotion 到正式 root 输出。
