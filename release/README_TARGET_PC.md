# ChemTsCorr Windows 目标电脑验收指南

本指南面向不安装开发工具的验收人员。发布物是 PyInstaller `onedir` 目录，不是安装程序。

## 准备

1. 复制整个发布目录，必须保留 `ChemTsCorr\` 及其全部文件；不要只复制 `ChemTsCorr.exe`。
2. 不要安装 Python、Git、Visual Studio 或 Conda，也不要修改发布目录内部文件。
3. 解压后先核对 ZIP 同目录 `.zip.sha256`。PowerShell 示例：

   ```powershell
   Get-FileHash .\ChemTsCorr-<version>-windows-x64.zip -Algorithm SHA256
   ```

4. ZIP 内 `release_manifest.json` 的 `manifest_scope` 是 `archive-internal`，其 `zip_sha256` 必须为 `null`，这是避免自引用哈希的正常设计。最终 ZIP 哈希以外部 `.zip.sha256` 和 `release_manifest.final.json` 为准。

## 依赖说明

- Python 包、Python 解释器、XGBoost/SHAP/Excel 引擎和 PyInstaller 运行文件已随整个 `ChemTsCorr\` 目录分发。
- Windows 必须提供可用的系统组件以及 Microsoft Edge WebView2 Runtime。VC++ Runtime 是否需要外部安装，以目标机实际启动结果为准，不能只看注册表。
- 如果提示缺少 `VCRUNTIME140.dll`、`VCRUNTIME140_1.dll`、`MSVCP140.dll` 或 `concrt140.dll`，由用户或 IT 管理员从 Microsoft 官方渠道手工安装受支持的 Visual C++ Redistributable；程序不会自动下载或安装。
- 如果 WebView2 缺失或损坏，由用户或 IT 管理员从 Microsoft 官方渠道手工安装/修复 Evergreen WebView2 Runtime；程序不会静默联网下载。

## 启动

双击 `ChemTsCorr\ChemTsCorr.exe`，等待桌面窗口出现。应用内部服务只监听 `127.0.0.1` 并使用动态端口；不要手工访问或对外开放该端口。

日志位置：`%LOCALAPPDATA%\ChemTsCorr\logs\desktop-launcher.log`。

## 自动验收

在解压目录打开普通 Windows PowerShell，不需要管理员权限：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\target_pc_acceptance.ps1
```

常用参数：

```powershell
.\target_pc_acceptance.ps1 -SkipLargeData
.\target_pc_acceptance.ps1 -SkipDefenderCheck
.\target_pc_acceptance.ps1 -SmallDataTimeoutMinutes 5 -LargeDataTimeoutMinutes 30
```

脚本会生成 JSON、Markdown 和日志三种报告。关键自动测试失败时返回非零退出码。Defender 命令不可用或被企业策略阻止时只能记录为 `Skipped`。

自动覆盖：

- 普通、空格、中文、中文加空格路径；
- 桌面窗口、本地首页 HTTP 200、上传、列识别、主筛查、下载；
- CSV、TXT、TSV、XLSX 和中文列名样例；
- 正常关闭、端口释放、连续启动 10 次、双实例、强制进程树关闭及恢复；
- 固定冒烟端口 8765 冲突；
- 40×45000 大数据主筛查（除非显式跳过）；
- 发布目录未写入用户数据、服务只监听回环地址；
- Defender 发布目录扫描（系统支持时）。

用户数据应只出现在：

```text
%LOCALAPPDATA%\ChemTsCorr\
  logs\
  uploads\
  web_runs\
```

## 必须人工执行的验收

自动脚本不会伪造 GUI 或特殊系统场景。逐项在 `acceptance_report_template.md` 填写真实结果：

1. 确认目标电脑未安装 Python 和开发工具，然后启动应用。
2. 分别上传 CSV、TXT、TSV、XLSX，以及另行准备的合法 XLS、XLSM 文件；确认时间列、数值列和中文目标变量可选择。
3. 完成小数据主筛查，打开结果页并下载至少一个文件；用对应程序确认下载文件可打开、CSV 中文编码正常。
4. 对大数据记录上传、列识别、主筛查耗时、峰值内存、平均 CPU、结果总大小和页面响应情况。
5. 以较少候选变量分别实际运行 Granger、SHAP 和 XGBoost；不得只根据主筛查推断通过。
6. 关闭窗口后检查主进程、`--desktop-service` 子进程和监听端口均消失；同时启动两个实例，确认互不影响且运行目录不覆盖。
7. 用任务管理器只结束主进程；在上传、分析、下载过程中关闭窗口；结束子服务；随后确认无长期残留且可以再次启动。
8. 人工验证用户目录不可写、磁盘空间不足、输出文件被占用，以及注销/关机前关闭。不要为了测试自动填满磁盘。
9. 在缺少 VC++ Runtime、缺少/损坏 WebView2 的隔离目标机执行真实启动并记录完整错误和日志。
10. 对发布目录和最终 ZIP 执行 Defender 或企业杀毒扫描。不得关闭杀毒软件、添加排除项或自动上传第三方服务。

未执行项目填写 `Not Tested`，不能填写 `Pass`。

## 问题排查

- 无窗口或页面空白：查看启动日志，记录 WebView2 状态、EXE 版本和 SHA-256。
- VC++ DLL 缺失：记录缺失 DLL 的完整名称，由 IT 手工安装 Microsoft 官方运行库。
- 杀毒软件拦截：不要绕过；复制 `false_positive_report_template.md`，记录检测名、引擎/病毒库版本和哈希，由发布负责人手工申诉。
- 上传或分析失败：记录输入格式、大小、所选列、任务状态和日志；不要把原始生产数据放入诊断包。
- 进程残留或端口冲突：记录 PID、命令行和监听地址，确认未监听 `0.0.0.0` 或局域网地址。
- 用户目录不可写：确认 `%LOCALAPPDATA%\ChemTsCorr\` 权限和剩余磁盘空间。

## 收集脱敏诊断包

```powershell
.\collect_diagnostics.ps1
```

脚本在压缩前列出将收集的文件，输出 `ChemTsCorr-diagnostics-<timestamp>.zip`。它只收集 ChemTsCorr 日志、最近运行的 `run_config.json`/`summary.md`、系统/运行库信息、相关进程端口和近期应用错误事件，并对密钥与用户路径脱敏。

默认排除原始上传数据、完整分析表、API Key、Authorization、Token、密码、LLM 密钥和其他应用日志。需要保留完整本机路径时必须显式使用 `-IncludeFullPaths`，并在对外发送前人工检查。
