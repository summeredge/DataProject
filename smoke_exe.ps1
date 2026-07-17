[CmdletBinding()]
param(
    [string]$ExePath = (Join-Path $PSScriptRoot 'dist\ChemTsCorr\ChemTsCorr.exe'),
    [int]$Port = 8765
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path $ExePath)) { throw "EXE not found: $ExePath" }

function Wait-ForUrl([string]$Url) {
    $deadline = (Get-Date).AddSeconds(30)
    do {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 1
            if ($response.StatusCode -eq 200) { return }
        } catch { }
        Start-Sleep -Milliseconds 250
    } while ((Get-Date) -lt $deadline)
    throw "Timed out waiting for $Url"
}

function Test-NormalDesktop {
    $desktop = Start-Process -FilePath $ExePath -PassThru
    try {
        Start-Sleep -Seconds 3
        if ($desktop.HasExited) { throw 'Desktop EXE exited before opening its window.' }
        $child = Get-CimInstance Win32_Process -Filter "ParentProcessId=$($desktop.Id)" |
            Where-Object { $_.CommandLine -match '--desktop-service' }
        if (-not $child) { throw 'Desktop EXE did not create its local service child process.' }
        Write-Host "Normal desktop launch passed (PID $($desktop.Id), service PID $($child.ProcessId))."
    } finally {
        & taskkill /PID $desktop.Id /T /F | Out-Null
        Start-Sleep -Seconds 1
    }
}

& $ExePath --module-check
if ($LASTEXITCODE -ne 0) { throw 'Packaged module check failed.' }
Write-Host 'XGBoost, SHAP, openpyxl, and xlrd module checks passed.'

Test-NormalDesktop

$process = Start-Process -FilePath $ExePath -ArgumentList '--desktop-service', '--host', '127.0.0.1', '--port', $Port -PassThru
try {
    $baseUrl = "http://127.0.0.1:$Port"
    Wait-ForUrl "$baseUrl/"

    $samplePath = Join-Path ([System.IO.Path]::GetTempPath()) 'chem-ts-corr-smoke.csv'
    @'
timestamp,target,driver
2025-01-01 00:00:00,1,2
2025-01-01 01:00:00,2,3
2025-01-01 02:00:00,3,4
2025-01-01 03:00:00,4,5
2025-01-01 04:00:00,5,6
2025-01-01 05:00:00,6,7
'@ | Set-Content -Path $samplePath -Encoding utf8

    $upload = & curl.exe --silent --show-error --fail -F "file=@$samplePath" "$baseUrl/api/upload" | ConvertFrom-Json
    if (-not $upload.file_id) { throw 'Upload response did not contain file_id.' }
    $columns = Invoke-RestMethod -Uri "$baseUrl/api/columns?file_id=$($upload.file_id)&encoding=utf-8-sig"
    if ($columns.columns -notcontains 'target') { throw 'Columns response did not contain target.' }

    $analysis = & curl.exe --silent --show-error --fail `
        -F "file_id=$($upload.file_id)" -F 'encoding=utf-8-sig' -F 'time_column=timestamp' `
        -F 'target=target' -F 'max_lag=2' -F 'top_k=1' -F 'min_valid_ratio=0.7' `
        "$baseUrl/api/analyze" | ConvertFrom-Json
    if (-not $analysis.task_id) { throw 'Analysis response did not contain task_id.' }

    $deadline = (Get-Date).AddSeconds(60)
    do {
        Start-Sleep -Seconds 1
        $status = Invoke-RestMethod -Uri "$baseUrl/api/status?task_id=$($analysis.task_id)"
    } while ($status.status -eq 'running' -and (Get-Date) -lt $deadline)
    if ($status.status -ne 'done') { throw "Analysis did not complete: $($status.error)" }

    $result = Invoke-RestMethod -Uri "$baseUrl/api/result?task_id=$($analysis.task_id)"
    # The result supplies a whitelisted /download URL for each generated artifact.
    $download = $result.downloads | Select-Object -First 1
    if (-not $download.url) { throw 'Analysis result did not contain a download.' }
    $downloadPath = Join-Path ([System.IO.Path]::GetTempPath()) 'chem-ts-corr-smoke-download'
    Invoke-WebRequest -UseBasicParsing -Uri "$baseUrl$($download.url)" -OutFile $downloadPath
    if (-not (Test-Path $downloadPath)) { throw 'Result download failed.' }
    Write-Host 'Upload, columns, analysis, result query, and download checks passed.'
} finally {
    if (-not $process.HasExited) { & taskkill /PID $process.Id /T /F | Out-Null }
    Remove-Item $samplePath, $downloadPath -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
}

if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $Port is still in use after the EXE was closed."
}
Write-Host 'Packaged service exited and its port was released.'
