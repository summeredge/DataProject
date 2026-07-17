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
            return [PSCustomObject]@{
                ProcessId = $child.ProcessId
                Port = [int]$Matches[1]
                Name = $child.Name
                CreationDate = $child.CreationDate
                CommandLine = $child.CommandLine
            }
        }
        Start-Sleep -Milliseconds 250
    } while ((Get-Date) -lt $deadline)
    throw 'Desktop EXE did not start a local service with a dynamic port within 30 seconds.'
}

function Wait-ForProcessExit([System.Diagnostics.Process]$Process, [string]$Description) {
    $deadline = (Get-Date).AddSeconds(10)
    do {
        if ($Process.HasExited) { return }
        Start-Sleep -Milliseconds 250
    } while ((Get-Date) -lt $deadline)
    throw "$Description (PID $($Process.Id)) did not exit."
}

function Wait-ForDesktopServiceExit($Service) {
    $deadline = (Get-Date).AddSeconds(10)
    do {
        $process = Get-CimInstance Win32_Process -Filter "ProcessId=$($Service.ProcessId)" -ErrorAction SilentlyContinue
        if (-not $process -or $process.Name -ne $Service.Name -or $process.CreationDate -ne $Service.CreationDate -or $process.CommandLine -ne $Service.CommandLine) {
            return
        }
        Start-Sleep -Milliseconds 250
    } while ((Get-Date) -lt $deadline)
    throw "Desktop service (PID $($Service.ProcessId)) did not exit."
}

function Test-NormalDesktop {
    $desktop = Start-Process -FilePath $ExePath -PassThru
    $service = $null
    try {
        $service = Wait-ForDesktopService $desktop
        $baseUrl = "http://127.0.0.1:$($service.Port)"
        Wait-ForUrl "$baseUrl/"
        if (-not $desktop.CloseMainWindow()) {
            throw 'User-style desktop window close request was rejected.'
        }
        Write-Host 'User-style desktop window close request succeeded.'
        Wait-ForProcessExit $desktop 'Desktop main process'
        Write-Host 'Desktop main process exited normally.'
        Wait-ForDesktopServiceExit $service
        Write-Host 'Desktop service exited normally.'
        if (Get-NetTCPConnection -LocalPort $service.Port -State Listen -ErrorAction SilentlyContinue) {
            throw "Desktop service port $($service.Port) is still in use after normal desktop shutdown."
        }
        Write-Host "Desktop service dynamic port $($service.Port) was released."
    } finally {
        if (-not $desktop.HasExited) {
            Write-Host 'Desktop normal shutdown failed; forced process-tree cleanup executed.'
            & taskkill /PID $desktop.Id /T /F | Out-Null
            Wait-ForProcessExit $desktop 'Desktop main process after forced cleanup'
        }
    }
}

& $ExePath --module-check
if ($LASTEXITCODE -ne 0) { throw 'Packaged module check failed.' }
Write-Host 'XGBoost, SHAP, openpyxl, and xlrd module checks passed.'

Test-NormalDesktop

$process = $null
$samplePath = $null
$downloadPath = $null
try {
    $process = Start-Process -FilePath $ExePath -ArgumentList '--desktop-service', '--host', '127.0.0.1', '--port', $Port -PassThru
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
    if ($process -and -not $process.HasExited) {
        if ($process.CloseMainWindow()) {
            try {
                Wait-ForProcessExit $process 'Internal service process'
            } catch {
                Write-Host 'Internal service did not stop normally; forced process-tree cleanup executed.'
                & taskkill /PID $process.Id /T /F | Out-Null
                Wait-ForProcessExit $process 'Internal service process after forced cleanup'
            }
        } else {
            Write-Host 'Internal service has no normal window shutdown; forced process-tree cleanup executed.'
            & taskkill /PID $process.Id /T /F | Out-Null
            Wait-ForProcessExit $process 'Internal service process after forced cleanup'
        }
    }
    if ($samplePath) {
        Remove-Item $samplePath -Force -ErrorAction SilentlyContinue
    }
    if ($downloadPath) {
        Remove-Item $downloadPath -Force -ErrorAction SilentlyContinue
    }
}

if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $Port is still in use after the EXE was closed."
}
Write-Host 'Packaged service exited and its port was released.'
