# ChemTsCorr 目标电脑验收报告

> 未实际执行的项目必须填写 `Not Tested`，不得填写 `Pass`。

## 基本信息

| 字段 | 值 |
| --- | --- |
| 目标机编号（不要填写机器名） | |
| Windows 版本 / Build | |
| CPU / 内存 | |
| WebView2 状态 | |
| VC++ Runtime 状态 | |
| 杀毒软件 | |
| 发布版本 | |
| Commit SHA | |
| EXE SHA-256 | |
| ZIP SHA-256 | |
| 测试日期 | |
| 测试人员 | |

## 测试矩阵

| 测试项 | 结果 | 备注/证据 |
| --- | --- | --- |
| 无 Python 启动 | Not Tested | |
| 无开发工具启动 | Not Tested | |
| VC++ 缺失场景 | Not Tested | |
| WebView2 缺失场景 | Not Tested | |
| 中文路径 | Not Tested | |
| 空格路径 | Not Tested | |
| CSV 上传 | Not Tested | |
| TXT 上传 | Not Tested | |
| TSV 上传 | Not Tested | |
| XLSX 上传 | Not Tested | |
| XLS/XLSM 上传 | Not Tested | 需要提供合法样例文件 |
| 中文列名 | Not Tested | |
| 40×45000 大数据分析 | Not Tested | |
| Granger | Not Tested | |
| SHAP | Not Tested | |
| XGBoost | Not Tested | |
| Defender 目录与 ZIP 扫描 | Not Tested | |
| 正常关闭 | Not Tested | |
| 异常关闭 | Not Tested | |
| 重复启动 | Not Tested | |
| 同时启动 | Not Tested | |
| 端口释放 / 仅绑定 127.0.0.1 | Not Tested | |
| 日志生成 | Not Tested | |
| 脱敏诊断包生成 | Not Tested | |

## 性能记录

记录上传、列识别、主筛查耗时，峰值内存、平均 CPU、结果总大小、参数和最终状态。低配置机器超时应记录硬件与参数，不直接判定为程序错误。

## 人工异常场景

记录用户目录不可写、磁盘空间不足、文件被占用、上传/分析/下载过程中关闭窗口、注销或关机前关闭等人工步骤与真实结果。
