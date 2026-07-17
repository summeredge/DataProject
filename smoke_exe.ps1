[CmdletBinding()]
param(
    [string]$ExePath = (Join-Path $PSScriptRoot 'dist\ChemTsCorr\ChemTsCorr.exe')
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path $ExePath)) { throw "EXE not found: $ExePath" }

function Wait-Homepage([string]$BaseUrl, [int]$TimeoutSeconds = 30) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        try {
            if ((Invoke-WebRequest -UseBasicParsing -Uri "$BaseUrl/" -TimeoutSec 1).StatusCode -eq 200) { return }
        } catch {}
        Start-Sleep -Milliseconds 250
    } while ((Get-Date) -lt $deadline)
    throw "Local homepage did not become available: $BaseUrl"
}

function Stop-ProcessTree([System.Diagnostics.Process]$Process) {
    if (-not $Process.HasExited) { & taskkill /PID $Process.Id /T /F | Out-Null }
}

function Assert-PortReleased([int]$Port) {
    if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
        throw "Port $Port is still listening after shutdown."
    }
}

function Test-PackagedService {
    $port = 8765
    $baseUrl = "http://127.0.0.1:$port"
    $process = Start-Process -FilePath $ExePath -ArgumentList '--desktop-service', '--host', '127.0.0.1', '--port', $port -PassThru
    try {
        Wait-Homepage $baseUrl
        $csv = Join-Path ([IO.Path]::GetTempPath()) 'chem-ts-corr-smoke.csv'
        @"
time,target,feed
2024-01-01 00:00,1,10
2024-01-01 00:01,2,11
2024-01-01 00:02,3,12
2024-01-01 00:03,4,13
2024-01-01 00:04,5,14
2024-01-01 00:05,6,15
2024-01-01 00:06,7,16
2024-01-01 00:07,8,17
2024-01-01 00:08,9,18
2024-01-01 00:09,10,19
2024-01-01 00:10,11,20
2024-01-01 00:11,12,21
"@ | Set-Content -Path $csv -Encoding utf8
        try {
            $upload = (& curl.exe -sS -X POST -F "file=@$csv;filename=smoke.csv" "$baseUrl/api/upload" | ConvertFrom-Json)
            $columns = Invoke-RestMethod -Uri "$baseUrl/api/columns?file_id=$($upload.file_id)&encoding=utf-8-sig"
            if ($columns.columns -notcontains 'target') { throw 'Uploaded CSV columns were not returned.' }
            $task = (& curl.exe -sS -X POST -F "file_id=$($upload.file_id)" -F 'encoding=utf-8-sig' -F 'time_column=time' -F 'target=target' -F 'max_lag=1' -F 'top_k=1' "$baseUrl/api/analyze" | ConvertFrom-Json)
            $deadline = (Get-Date).AddSeconds(60)
            do {
                $status = Invoke-RestMethod -Uri "$baseUrl/api/status?task_id=$($task.task_id)"
                if ($status.status -eq 'error') { throw $status.error }
                Start-Sleep -Milliseconds 250
            } while ($status.status -ne 'done' -and (Get-Date) -lt $deadline)
            if ($status.status -ne 'done') { throw 'Minimal screening did not finish.' }
            $result = Invoke-RestMethod -Uri "$baseUrl/api/result?task_id=$($task.task_id)"
            $runDir = Join-Path $env:LOCALAPPDATA "ChemTsCorr\web_runs\$($result.run_id)"
            $download = Join-Path ([IO.Path]::GetTempPath()) 'chem-ts-corr-ranked-features.csv'
            Invoke-WebRequest -UseBasicParsing -Uri "$baseUrl/download?run_id=$($result.run_id)&file=ranked_features.csv" -OutFile $download
            if (-not (Test-Path (Join-Path $runDir 'ranked_features.csv')) -or -not (Test-Path $download)) {
                throw 'Screening result or download was not generated.'
            }
        } finally { Remove-Item $csv, $download -Force -ErrorAction SilentlyContinue }
        Write-Host 'Internal service, upload, columns, screening, results, and download checks passed.'
    } finally {
        Stop-ProcessTree $process
        Start-Sleep -Seconds 1
    }
    Assert-PortReleased $port
}

function Test-NormalDesktop {
    $desktop = Start-Process -FilePath $ExePath -PassThru
    $service = $null
    try {
        $deadline = (Get-Date).AddSeconds(30)
        do {
            $service = Get-CimInstance Win32_Process -Filter "ParentProcessId = $($desktop.Id)" | Select-Object -First 1
            if ($service) {
                $connection = Get-NetTCPConnection -OwningProcess $service.ProcessId -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
                if ($connection) { break }
            }
            Start-Sleep -Milliseconds 250
        } while ((Get-Date) -lt $deadline)
        if (-not $connection) { throw 'Desktop mode did not start a listening child service.' }
        Wait-Homepage "http://127.0.0.1:$($connection.LocalPort)"
        Write-Host "Normal desktop mode started child service on port $($connection.LocalPort)."
    } finally {
        Stop-ProcessTree $desktop
        Start-Sleep -Seconds 1
    }
    if ($service -and (Get-Process -Id $service.ProcessId -ErrorAction SilentlyContinue)) {
        throw 'Desktop child service remained after the desktop process was closed.'
    }
    if ($connection) { Assert-PortReleased $connection.LocalPort }
}

& $ExePath --module-check
if ($LASTEXITCODE -ne 0) { throw 'Packaged optional module import check failed.' }
Write-Host 'Packaged statsmodels, scikit-learn, SHAP, and XGBoost imports passed.'
Test-PackagedService
Test-NormalDesktop
Write-Host 'All packaged EXE smoke tests passed.'
