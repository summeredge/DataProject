from pathlib import Path


def function_body(script: str, name: str, next_name: str) -> str:
    return script.split(f"function {name}", 1)[1].split(f"function {next_name}", 1)[0]


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


def test_smoke_script_waits_for_real_desktop_main_window_before_closing():
    smoke_script = Path("smoke_exe.ps1").read_text(encoding="utf-8")
    main_window_wait = function_body(smoke_script, "Wait-ForMainWindow", "Wait-ForDesktopService")

    assert "$Process.HasExited" in main_window_wait
    assert "$Process.Refresh()" in main_window_wait
    assert "$Process.MainWindowHandle -ne [IntPtr]::Zero" in main_window_wait
    assert "Desktop EXE exited before creating its main window." in main_window_wait
    assert "Desktop main window was not created within $TimeoutSeconds seconds." in main_window_wait
    assert "Start-Sleep -Milliseconds 250" in main_window_wait

    normal_desktop = smoke_script.split("function Test-NormalDesktop", 1)[1].split("& $ExePath --module-check", 1)[0]
    normal_shutdown, normal_finally = normal_desktop.split("} finally {", 1)
    assert normal_shutdown.index('Wait-ForUrl "$baseUrl/"') < normal_shutdown.index("Wait-ForMainWindow $desktop")
    assert normal_shutdown.index("Wait-ForMainWindow $desktop") < normal_shutdown.index("CloseMainWindow")
    assert "Desktop main window became ready (handle $($desktop.MainWindowHandle))." in normal_shutdown
    assert "Start-Sleep -Seconds 3" not in normal_shutdown
    assert "taskkill" not in normal_shutdown
    assert "Wait-ForProcessExit $desktop 'Desktop main process'" in normal_shutdown
    assert "Wait-ForDesktopServiceExit $service" in normal_shutdown
    assert "Desktop service dynamic port $($service.Port) was released" in normal_shutdown


def test_desktop_fallback_cleanup_distinguishes_service_states_and_preserves_failure():
    smoke_script = Path("smoke_exe.ps1").read_text(encoding="utf-8")

    raw_lookup = function_body(smoke_script, "Get-DesktopServiceProcessById", "Test-DesktopServiceIdentity")
    assert 'Get-CimInstance Win32_Process -Filter "ProcessId=$($Service.ProcessId)"' in raw_lookup

    identity_check = function_body(smoke_script, "Test-DesktopServiceIdentity", "Get-MatchingDesktopService")
    for field in ("ProcessId", "Name", "CreationDate", "CommandLine"):
        assert f"$Process.{field}" in identity_check
        assert f"$Service.{field}" in identity_check

    matcher = function_body(smoke_script, "Get-MatchingDesktopService", "Wait-ForDesktopServiceExit")
    assert "Get-DesktopServiceProcessById $Service" in matcher
    assert "Test-DesktopServiceIdentity $Service $process" in matcher

    service_exit = function_body(smoke_script, "Wait-ForDesktopServiceExit", "Test-NormalDesktop")
    assert "Get-MatchingDesktopService $Service" in service_exit

    normal_desktop = smoke_script.split("function Test-NormalDesktop", 1)[1].split("& $ExePath --module-check", 1)[0]
    normal_shutdown, normal_finally = normal_desktop.split("} finally {", 1)
    service_cleanup = normal_finally.split("if ($service) {", 1)[1].split("try {\n                if (Get-NetTCPConnection", 1)[0]
    missing_branch, remaining = service_cleanup.split("} elseif (Test-DesktopServiceIdentity $service $serviceProcess) {", 1)
    matched_branch, mismatch_branch = remaining.split("} else {", 1)

    assert "Get-DesktopServiceProcessById $service" in service_cleanup
    assert "Desktop service already exited; no fallback cleanup was required." in missing_branch
    assert "taskkill" not in missing_branch
    assert "Leftover desktop service detected; executing identity-verified fallback cleanup." in matched_branch
    assert "& taskkill /PID $service.ProcessId /T /F" in matched_branch
    assert "Wait-ForDesktopServiceExit $service" in matched_branch
    assert "Desktop service PID was reused or its identity changed; skipping cleanup." in mismatch_branch
    assert "taskkill" not in mismatch_branch
    assert normal_finally.index("if ($service)") > normal_finally.index("if (-not $desktop.HasExited)")
    assert "$originalFailure = $_" in normal_desktop
    assert "if ($originalFailure) { throw $originalFailure }" in normal_desktop
    assert "if ($cleanupFailure) { throw $cleanupFailure }" in normal_desktop
