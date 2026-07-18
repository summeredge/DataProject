import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release"


def read_release_file(name: str) -> str:
    return (RELEASE / name).read_text(encoding="utf-8")


def test_target_pc_release_files_and_manifest_contract_exist():
    required = {
        "README_TARGET_PC.md",
        "target_pc_acceptance.ps1",
        "collect_diagnostics.ps1",
        "generate_acceptance_data.py",
        "acceptance_report_template.md",
        "release_manifest_template.json",
        "false_positive_report_template.md",
    }

    assert required <= {path.name for path in RELEASE.iterdir() if path.is_file()}

    manifest = json.loads(read_release_file("release_manifest_template.json"))
    assert manifest["product"] == "ChemTsCorr"
    assert manifest["architecture"] == "windows-x64"
    assert manifest["package_type"] == "onedir"
    assert manifest["manifest_scope"] == "archive-internal"
    assert manifest["zip_sha256"] is None
    assert {
        "pytest",
        "static_packaging_tests",
        "ruff",
        "build_exe",
        "smoke_exe",
        "target_pc",
    } <= manifest["tests"].keys()


def test_target_pc_script_has_real_checks_and_no_development_runtime_dependency():
    script = read_release_file("target_pc_acceptance.ps1")

    for marker in (
        "Get-CimInstance Win32_OperatingSystem",
        "Get-MpComputerStatus",
        "Get-NetTCPConnection",
        "Invoke-WebRequest",
        "System.Net.Http.MultipartFormDataContent",
        "127.0.0.1",
        "Program Test",
        "$chinesePath",
        "$chineseSpacePath",
        "Sequential restart 10 times",
        "Concurrent instances",
        "Port release",
        "ExclusiveAddressUse = $true",
        "User data directories are writable and outside the release",
        "Original release directory remained unchanged",
        "ConvertTo-Json",
        "exit 1",
    ):
        assert marker in script

    lowered = script.lower()
    assert not re.search(r"(?im)^\s*(?:&\s*)?python(?:\.exe)?\s", script)
    assert "get-command python" not in lowered
    assert not re.search(r"(?im)^\s*(?:&\s*)?pip(?:\.exe)?\s", script)
    assert not re.search(r"(?im)^\s*(?:&\s*)?git(?:\.exe)?\s", script)
    assert "set-mppreference" not in lowered
    assert "add-mppreference" not in lowered
    assert "virustotal" not in lowered
    assert "function Wait-ForExit([int]$ProcessId" in script
    assert "function Wait-ForExit([int]$Pid" not in script
    assert "Join-Path $PSScriptRoot 'acceptance-results'" not in script
    assert "ChemTsCorr\\acceptance-results" in script


def test_diagnostics_is_allowlisted_and_redacts_secrets():
    script = read_release_file("collect_diagnostics.ps1")

    for marker in (
        "ChemTsCorr-diagnostics-",
        "Get-WinEvent",
        "Get-FileHash",
        "run_config.json",
        "summary.md",
        "Files to be collected",
        "api[_-]?key",
        "authorization",
        "token",
        "password",
    ):
        assert marker.lower() in script.lower()

    assert "join-path $userdata 'uploads'" not in script.lower()
    assert "Compress-Archive" in script


def test_package_script_uses_non_circular_archive_hash_contract():
    script = (ROOT / "package_release.ps1").read_text(encoding="utf-8")

    assert script.index("release_manifest.json") < script.index("Compress-Archive")
    assert ".zip.sha256" in script
    assert script.index("Compress-Archive") < script.index("Set-Content -LiteralPath $ZipHashPath")
    assert "release_manifest.final.json" in script
    assert 'manifest_scope = "external-release"' in script
    assert "archive_name" in script
    assert "zip_sha256 = $ZipHash" in script
    assert "SHA256SUMS.txt" in script
    assert "build_exe.ps1" in script
    assert "smoke_exe.ps1" in script
    assert "generate_acceptance_data.py" in script
    assert "Get-MpComputerStatus -ErrorAction Stop" in script
    assert "Start-MpScan -ScanType CustomScan -ScanPath $PackageRoot -ErrorAction Stop" in script
    assert "Start-MpScan -ScanType CustomScan -ScanPath $ZipPath -ErrorAction Stop" in script
    assert "Skipped: $DefenderUnavailableReason" in script


def test_acceptance_data_generator_is_deterministic_and_covers_supported_formats(tmp_path):
    generator = RELEASE / "generate_acceptance_data.py"
    args = [
        sys.executable,
        str(generator),
        "--output-dir",
        str(tmp_path),
        "--large-rows",
        "300",
        "--large-variables",
        "8",
    ]
    subprocess.run(args, check=True)
    first_hash = (tmp_path / "acceptance_large.csv").read_bytes()
    subprocess.run(args, check=True)

    assert first_hash == (tmp_path / "acceptance_large.csv").read_bytes()
    assert {
        "acceptance_small.csv",
        "acceptance_small.txt",
        "acceptance_small.tsv",
        "acceptance_small.xlsx",
        "acceptance_chinese_columns.csv",
        "acceptance_chinese_columns.xlsx",
        "acceptance_large.csv",
    } <= {path.name for path in tmp_path.iterdir()}

    small = pd.read_csv(tmp_path / "acceptance_small.csv", encoding="utf-8-sig")
    chinese = pd.read_excel(tmp_path / "acceptance_chinese_columns.xlsx")
    large = pd.read_csv(tmp_path / "acceptance_large.csv", encoding="utf-8-sig")

    assert len(small) >= 200
    assert small.columns.tolist() == ["timestamp", "target", "driver_1", "driver_2", "noise"]
    assert small["timestamp"].duplicated().any()
    assert small.isna().sum().sum() > 0
    assert chinese.columns.tolist() == ["时间", "目标变量", "温度", "压力", "流量"]
    assert large.shape == (300, 9)
