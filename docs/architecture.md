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

当前实际执行模式：

```text
raw
detrend
diff
detrend_diff
```

契约新模式（仅定义，暂未实现，不得进入执行流程）：

```text
lowpass
lowpass_detrend
lowpass_diff
```

模式语义与配置字段约束见 `docs/contracts.md`。`lowpass*` 模式可在配置对象中表示，
但执行前必须被明确拒绝，不得静默回退到 `raw`。

## 初筛双分支（规划中）

非 Raw 初筛将按两个独立分支分别运行第一阶段，产物存放在独立子目录：

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

当前阶段只冻结目录与状态契约，不实现双分支执行、分支确认或对比文件生成。
