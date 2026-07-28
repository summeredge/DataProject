# Changelog

<!-- SCHEMA: {"ts":"ISO-8601","action":"add|promote|extract|resolve","type":"learning|error|feature","id":"entry ID","summary":"<=100 chars","target":"promotion target (optional)"} -->

```jsonl
{"ts":"2026-07-27T00:00:01+08:00","action":"add","type":"error","id":"ERR-20260727-002","summary":"受限会话拒绝启动 Python 时需请求真实 Windows 执行路径"}
{"ts":"2026-07-27T00:00:00+08:00","action":"add","type":"error","id":"ERR-20260727-001","summary":"PowerShell 搜索不要使用 Unix 文件通配路径"}
{"ts":"2026-07-20T00:00:00+08:00","action":"add","type":"error","id":"ERR-20260720-001","summary":"工况筛选必须在完整时间轴构造滞后后按目标时刻应用"}
{"ts":"2026-07-20T00:00:01+08:00","action":"resolve","type":"error","id":"ERR-20260720-001","summary":"XGB 验证已统一为完整时间轴构造滞后并按目标时刻筛选工况"}
```
{"date":"2026-07-27","type":"error","id":"ERR-20260727-003","summary":"组合回归测试超时，后续拆分小批次执行"}
{"date":"2026-07-27","type":"error","id":"ERR-20260727-004","summary":"含稳定性与基线重算的完整合成回归超时，改为拆分执行"}
{"date":"2026-07-27","type":"error","id":"ERR-20260727-005","summary":"完整 pytest 暴露 24 个与新初筛契约冲突的旧断言"}
{"date":"2026-07-27","type":"error","id":"ERR-20260727-006","summary":"大批量受影响测试组合因 stdout/超时异常未返回汇总，后续拆分执行"}
{"date":"2026-07-27","type":"error","id":"ERR-20260727-007","summary":"完整回归剩余一条稳定性字段旧断言，已迁移为初筛不暴露契约"}
