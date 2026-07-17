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

脚本会检查 Python、PyInstaller 及 `scikit-learn`、`statsmodels`、`matplotlib`、`shap`、`xgboost` 和 `pywebview`，清理旧 `build/` 与 `dist/ChemTsCorr/`，并使用 `ChemTsCorr.spec` 构建。构建后会确认 EXE 和 XGBoost 原生动态库存在，并输出发布目录大小；任一关键文件缺失都会返回非零退出码。

发布目录为 `dist\ChemTsCorr\`；分发整个目录，不能只复制 `ChemTsCorr.exe`。双击 `ChemTsCorr.exe` 启动。

## 数据与日志

开发模式继续把 Web 上传和分析结果写到项目内的 `reports/`。打包 EXE 则写到 `%LOCALAPPDATA%\ChemTsCorr\`：上传在 `uploads/`，分析结果在 `web_runs/`，启动日志在 `logs\desktop-launcher.log`。打包目录和 PyInstaller 的 `_MEIPASS` 只用于只读程序资源。

## 构建后冒烟测试

关闭其他占用 8765 端口的本地应用后执行：

```powershell
.\smoke_exe.ps1
```

该脚本不依赖系统 Python。它先以 EXE 内部服务模式检查可选模块导入、上传 CSV、字段读取、最小主筛查、结果目录、下载接口和端口释放；随后不带参数启动正常桌面模式，检查桌面子服务的动态监听端口与首页，并通过进程树关闭后确认子进程和端口已释放。正常用户启动不需要任何参数。

## 目标电脑人工验收清单

在一台**未安装 Python** 的 Windows 电脑上，将整个 `dist\ChemTsCorr\` 复制到中文、含空格且非项目目录的位置后，逐项确认：

* 双击 `ChemTsCorr.exe` 不显示命令行窗口或额外浏览器，桌面窗口加载现有页面。
* CSV 和 XLSX 均可上传；完成主筛查后可查看趋势图并下载结果。
* Granger、模型解释（scikit-learn/SHAP）和 XGBoost 功能可启动并给出结果或既有明确提示。
* `%LOCALAPPDATA%\ChemTsCorr\` 中生成上传、结果和日志；发布目录没有新增用户数据。
* 关闭窗口后没有 `ChemTsCorr.exe` 子进程，监听端口已释放。

### WebView2

Windows 的 pywebview 后端通常需要 Microsoft Edge WebView2 Runtime。缺失时桌面窗口可能无法创建或启动后立即显示错误窗口；安装与系统架构匹配的 [Microsoft Edge WebView2 Runtime](https://developer.microsoft.com/microsoft-edge/webview2/) 后重新启动。若仍失败，请一并收集 `%LOCALAPPDATA%\ChemTsCorr\logs\desktop-launcher.log`。

## 常见错误

* **缺少模块：** 重新执行上述完整依赖安装命令；EXE 不会在运行时执行 `pip install`。
* **启动失败：** 查看 `%LOCALAPPDATA%\ChemTsCorr\logs\desktop-launcher.log`。开发模式请查看系统临时目录下的 `chem-ts-corr\desktop-launcher.log`。
* **端口冲突：** 关闭占用 `127.0.0.1:8765` 的程序后重试。桌面启动器正常会自动选择可用端口；冒烟脚本为固定首页检查使用 8765。
