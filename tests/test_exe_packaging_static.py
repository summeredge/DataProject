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
    assert "Wait-ForProcessExit $service.ProcessId" in smoke_script
    assert "Desktop service port $($service.Port) is still in use" in smoke_script
