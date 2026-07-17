# ChemTsCorr Windows EXE 打包（onedir 测试版）

这是免安装的 `onedir` 测试版，不是 MSI 或正式安装程序。应用会在本机启动一个仅绑定到 `127.0.0.1` 的服务，再用桌面窗口显示现有页面；不会打开额外浏览器或控制台窗口。

## 构建环境

在 Windows 上使用 Python 3.10+，并从项目根目录安装完整分析和打包依赖：

```powershell
python -m pip install -e ".[full,xgb]"
python -m pip install pyinstaller
```

执行构建：

```powershell
.\build_exe.ps1
```

脚本会检查 Python、PyInstaller 及 `scikit-learn`、`statsmodels`、`matplotlib`、`shap`、`xgboost`、`openpyxl`、`xlrd` 和 `pywebview`，清理旧 `build/` 与 `dist/ChemTsCorr/`，并使用 `ChemTsCorr.spec` 构建。构建后会确认发布目录存在 XGBoost DLL、启动 EXE 导入 XGBoost/SHAP/Excel 引擎，并输出 `Release size:`。任一步失败都会返回非零退出码。

发布目录为 `dist\ChemTsCorr\`；分发整个目录，不能只复制 `ChemTsCorr.exe`。双击 `ChemTsCorr.exe` 启动。

## 数据与日志

开发模式继续把 Web 上传和分析结果写到项目内的 `reports/`。打包 EXE 则写到 `%LOCALAPPDATA%\ChemTsCorr\`：上传在 `uploads/`，分析结果在 `web_runs/`，启动日志在 `logs\desktop-launcher.log`。打包目录和 PyInstaller 的 `_MEIPASS` 只用于只读程序资源。

## 构建后冒烟测试

关闭其他占用 8765 端口的本地应用后执行：

```powershell
.\smoke_exe.ps1
```

该脚本先验证 EXE 的 XGBoost、SHAP、`openpyxl` 和 `xlrd` 导入；随后无参数启动桌面模式，检查桌面主进程及其本地服务子进程。首页就绪后，自动测试会尝试通过关闭主窗口验证应用自身清理：确认主进程、子服务和动态端口均已退出或释放；`taskkill` 仅用于测试失败后的兜底清理。最后用内部服务模式执行上传、列识别、最小分析、结果查询和下载，并在关闭后检查 8765 端口已释放。正常用户启动不需要任何参数。最终仍需在真实 Windows 电脑上人工点击关闭按钮，确认没有残留进程。

## 人工验收清单

在一台**未安装 Python** 的 Windows 电脑上，将整个 `dist\ChemTsCorr\` 目录复制到包含中文和空格的路径（例如 `C:\测试 发布\ChemTsCorr`），然后逐项确认：

1. 双击 `ChemTsCorr.exe` 能打开桌面窗口；关闭窗口后，任务管理器中不再保留 `ChemTsCorr.exe` 或本地服务子进程。
2. 分别上传 CSV 和 XLSX 文件，能识别时间列及数值列，并完成主筛查。
3. 对上传数据分别运行 Granger、SHAP/模型解释和 XGBoost 验证，确认页面展示结果且下载区可下载生成的文件。
4. 若机器缺少 Microsoft Edge WebView2 Runtime，按系统提示安装 WebView2 Runtime 后重新启动；若仍失败，收集 `%LOCALAPPDATA%\ChemTsCorr\logs\desktop-launcher.log`。

## 常见错误

* **缺少模块：** 重新执行上述完整依赖安装命令；EXE 不会在运行时执行 `pip install`。
* **启动失败：** 查看 `%LOCALAPPDATA%\ChemTsCorr\logs\desktop-launcher.log`。开发模式请查看系统临时目录下的 `chem-ts-corr\desktop-launcher.log`。
* **端口冲突：** 关闭占用 `127.0.0.1:8765` 的程序后重试。桌面启动器正常会自动选择可用端口；冒烟脚本为固定首页检查使用 8765。
