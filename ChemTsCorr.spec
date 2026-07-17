# -*- mode: python ; coding: utf-8 -*-
"""Windows onedir build for the desktop launcher.

The web page is embedded in chem_ts_corr.web.INDEX_HTML, so this project has no
runtime template/static/config data directories to add to datas.
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


datas = []
hiddenimports = (
    collect_submodules("webview")
    + collect_submodules("sklearn")
    + collect_submodules("statsmodels")
    + collect_submodules("matplotlib")
    + collect_submodules("shap")
    + collect_submodules("xgboost")
)
datas += collect_data_files("matplotlib")
datas += collect_data_files("shap")
datas += collect_data_files("xgboost")

a = Analysis(
    ["chem_ts_corr/desktop.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ChemTsCorr",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    version="ChemTsCorr_version.txt",
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="ChemTsCorr",
)
