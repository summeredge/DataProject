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

脚本会检查 Python、PyInstaller 及 `scikit-learn`、`statsmodels`、`matplotlib`、`shap`、`xgboost` 和 `pywebview`，清理旧 `build/` 与 `dist/ChemTsCorr/`，并使用 `ChemTsCorr.spec` 构建。失败时会返回非零退出码。

发布目录为 `dist\ChemTsCorr\`；分发整个目录，不能只复制 `ChemTsCorr.exe`。双击 `ChemTsCorr.exe` 启动。

## 数据与日志

开发模式继续把 Web 上传和分析结果写到项目内的 `reports/`。打包 EXE 则写到 `%LOCALAPPDATA%\ChemTsCorr\`：上传在 `uploads/`，分析结果在 `web_runs/`，启动日志在 `logs\desktop-launcher.log`。打包目录和 PyInstaller 的 `_MEIPASS` 只用于只读程序资源。

## 构建后冒烟测试

关闭其他占用 8765 端口的本地应用后执行：

```powershell
.\smoke_exe.ps1
```

该脚本用 EXE 的内部服务模式检查 EXE 存在、启动进程、访问本地首页，并在关闭后检查 8765 端口已释放；正常用户启动不需要任何参数。

## 常见错误

* **缺少模块：** 重新执行上述完整依赖安装命令；EXE 不会在运行时执行 `pip install`。
* **启动失败：** 查看 `%LOCALAPPDATA%\ChemTsCorr\logs\desktop-launcher.log`。开发模式请查看系统临时目录下的 `chem-ts-corr\desktop-launcher.log`。
* **端口冲突：** 关闭占用 `127.0.0.1:8765` 的程序后重试。桌面启动器正常会自动选择可用端口；冒烟脚本为固定首页检查使用 8765。
