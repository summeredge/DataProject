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

function Wait-ForDesktopService([System.Diagnostics.Process]$Desktop) {
    $deadline = (Get-Date).AddSeconds(30)
    do {
        if ($Desktop.HasExited) { throw 'Desktop EXE exited before opening its window.' }
        $child = Get-CimInstance Win32_Process -Filter "ParentProcessId=$($Desktop.Id)" |
            Where-Object { $_.CommandLine -match '--desktop-service' } |
            Select-Object -First 1
        if ($child -and $child.CommandLine -match '--port\s+"?(\d+)"?') {
            return [PSCustomObject]@{ ProcessId = $child.ProcessId; Port = [int]$Matches[1] }
        }
        Start-Sleep -Milliseconds 250
    } while ((Get-Date) -lt $deadline)
    throw 'Desktop EXE did not start a local service with a dynamic port within 30 seconds.'
}

function Wait-ForProcessExit([int]$ProcessId) {
    $deadline = (Get-Date).AddSeconds(10)
    do {
        if (-not (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) { return }
        Start-Sleep -Milliseconds 250
    } while ((Get-Date) -lt $deadline)
    throw "Process $ProcessId did not exit after the desktop EXE was closed."
}

function Test-NormalDesktop {
    $desktop = Start-Process -FilePath $ExePath -PassThru
    $service = $null
    try {
        $service = Wait-ForDesktopService $desktop
        $baseUrl = "http://127.0.0.1:$($service.Port)"
        Wait-ForUrl "$baseUrl/"
        Write-Host "Normal desktop launch passed (PID $($desktop.Id), service PID $($service.ProcessId), port $($service.Port))."
    } finally {
        if (-not $desktop.HasExited) { & taskkill /PID $desktop.Id /T /F | Out-Null }
        Wait-ForProcessExit $desktop.Id
        if ($service) {
            Wait-ForProcessExit $service.ProcessId
            if (Get-NetTCPConnection -LocalPort $service.Port -State Listen -ErrorAction SilentlyContinue) {
                throw "Desktop service port $($service.Port) is still in use after the EXE was closed."
            }
        }
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
    @('timestamp,target,driver') + (0..49 | ForEach-Object {
        $timestamp = (Get-Date '2025-01-01 00:00:00').AddHours($_).ToString('yyyy-MM-dd HH:mm:ss')
        "$timestamp,$($_ + 1),$($_ + 2)"
    }) | Set-Content -Path $samplePath -Encoding utf8

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
