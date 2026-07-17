from pathlib import Path


def test_spec_collects_xgboost_native_libraries_and_excel_engines():
    spec = Path("ChemTsCorr.spec").read_text(encoding="utf-8")

    assert "collect_dynamic_libs" in spec
    assert 'binaries = collect_dynamic_libs("xgboost")' in spec
    assert "binaries=binaries" in spec
    for module in ("pandas.io.excel._openpyxl", "pandas.io.excel._xlrd", "openpyxl", "xlrd"):
        assert module in spec


def test_build_and_smoke_scripts_verify_packaged_requirements():
    build_script = Path("build_exe.ps1").read_text(encoding="utf-8")
    smoke_script = Path("smoke_exe.ps1").read_text(encoding="utf-8")

    assert "*xgboost*.dll" in build_script
    assert "Release size:" in build_script
    for marker in ("--module-check", "/api/upload", "/api/columns", "/api/analyze", "/download", "Test-NormalDesktop"):
        assert marker in smoke_script


def test_smoke_script_uses_sufficient_data_and_validates_desktop_readiness():
    smoke_script = Path("smoke_exe.ps1").read_text(encoding="utf-8")

    assert "0..49" in smoke_script
    assert "Wait-ForDesktopService" in smoke_script
    assert "Wait-ForUrl \"$baseUrl/\"" in smoke_script
    assert "CloseMainWindow" in smoke_script
    assert "Wait-ForProcessExit $desktop 'Desktop main process'" in smoke_script
    assert "Wait-ForDesktopServiceExit $service" in smoke_script
    assert "Desktop service dynamic port $($service.Port) was released" in smoke_script
    assert "$samplePath = $null" in smoke_script
    assert "$downloadPath = $null" in smoke_script

    normal_desktop = smoke_script.split("function Test-NormalDesktop", 1)[1].split("& $ExePath --module-check", 1)[0]
    normal_shutdown, normal_finally = normal_desktop.split("} finally {", 1)
    assert "CloseMainWindow" in normal_shutdown
    assert "Wait-ForProcessExit $desktop 'Desktop main process'" in normal_shutdown
    assert "Wait-ForDesktopServiceExit $service" in normal_shutdown
    assert "taskkill" not in normal_shutdown
    assert "taskkill" in normal_finally
