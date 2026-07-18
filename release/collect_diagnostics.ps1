[CmdletBinding()]
param(
    [string]$ExePath = (Join-Path $PSScriptRoot 'ChemTsCorr\ChemTsCorr.exe'),
    [string]$OutputPath = $PSScriptRoot,
    [switch]$IncludeFullPaths
)

$ErrorActionPreference = 'Stop'
$Timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$UserData = Join-Path $env:LOCALAPPDATA 'ChemTsCorr'
$OutputPath = [IO.Path]::GetFullPath($OutputPath)
New-Item -ItemType Directory -Path $OutputPath -Force | Out-Null
$Stage = Join-Path $env:TEMP "ChemTsCorr-diagnostics-$Timestamp"
$ZipPath = Join-Path $OutputPath "ChemTsCorr-diagnostics-$Timestamp.zip"

function Redact-Text([string]$Text) {
    if ($null -eq $Text) { return '' }
    $result = $Text -replace '(?im)("?(api[_-]?key|authorization|token|password|secret|llm[_-]?key)"?\s*[:=]\s*)[^,\r\n}]+', '$1"[REDACTED]"'
    if ($env:COMPUTERNAME) { $result = $result.Replace($env:COMPUTERNAME, '[REDACTED]') }
    if (-not $IncludeFullPaths) {
        if ($env:USERPROFILE) { $result = $result.Replace($env:USERPROFILE, '%USERPROFILE%') }
        if ($env:USERNAME) { $result = $result.Replace($env:USERNAME, '[REDACTED]') }
    }
    return $result
}

function Write-RedactedFile([string]$Source, [string]$Destination) {
    $text = Get-Content -LiteralPath $Source -Raw -ErrorAction Stop
    Redact-Text $text | Set-Content -LiteralPath $Destination -Encoding UTF8
}

function Get-WebView2Info {
    $paths = @(
        "${env:ProgramFiles(x86)}\Microsoft\EdgeWebView\Application",
        "$env:ProgramFiles\Microsoft\EdgeWebView\Application",
        "$env:LOCALAPPDATA\Microsoft\EdgeWebView\Application"
    ) | Where-Object { $_ -and (Test-Path $_) }
    [pscustomobject]@{ installed_paths = @($paths | ForEach-Object { Redact-Text $_ }) }
}

function Get-VcRuntimeInfo {
    $keys = @(
        'HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64',
        'HKLM:\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64'
    )
    @($keys | Where-Object { Test-Path $_ } | ForEach-Object {
        $value = Get-ItemProperty -Path $_ -ErrorAction SilentlyContinue
        [pscustomobject]@{ key = $_; installed = $value.Installed; version = $value.Version }
    })
}

if (Test-Path $Stage) { Remove-Item -LiteralPath $Stage -Recurse -Force }
New-Item -ItemType Directory -Path $Stage -Force | Out-Null
try {
    $metadata = [ordered]@{
        generated_at = (Get-Date).ToString('o')
        redacted = $true
        raw_uploads_included = $false
        windows = Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, BuildNumber, OSArchitecture
        disk = Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | Select-Object DeviceID, Size, FreeSpace
        webview2 = Get-WebView2Info
        vc_runtime = Get-VcRuntimeInfo
        exe = if (Test-Path -LiteralPath $ExePath) {
            $item = Get-Item -LiteralPath $ExePath
            [ordered]@{
                name = $item.Name
                path = Redact-Text $item.FullName
                file_version = $item.VersionInfo.FileVersion
                product_version = $item.VersionInfo.ProductVersion
                sha256 = (Get-FileHash -LiteralPath $ExePath -Algorithm SHA256).Hash
            }
        } else { [ordered]@{ error = 'EXE not found'; path = Redact-Text $ExePath } }
    }
    $metadata | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $Stage 'system-and-exe.json') -Encoding UTF8

    $processes = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -eq 'ChemTsCorr.exe' -or $_.CommandLine -match '--desktop-service' } |
        ForEach-Object {
            [pscustomobject]@{
                process_id = $_.ProcessId
                parent_process_id = $_.ParentProcessId
                name = $_.Name
                command_line = Redact-Text $_.CommandLine
                creation_date = $_.CreationDate
            }
        })
    $processes | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $Stage 'processes.json') -Encoding UTF8

    $processIds = @($processes | ForEach-Object process_id)
    $ports = if ($processIds) {
        @(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
            Where-Object { $_.OwningProcess -in $processIds } |
            Select-Object LocalAddress, LocalPort, OwningProcess, State)
    } else { @() }
    $ports | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $Stage 'listening-ports.json') -Encoding UTF8

    $logSource = Join-Path $UserData 'logs'
    if (Test-Path $logSource) {
        $logDest = Join-Path $Stage 'logs'
        New-Item -ItemType Directory -Path $logDest -Force | Out-Null
        Get-ChildItem -LiteralPath $logSource -File -Filter '*.log' -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending | Select-Object -First 10 | ForEach-Object {
                Write-RedactedFile $_.FullName (Join-Path $logDest $_.Name)
            }
    }

    $runs = Join-Path $UserData 'web_runs'
    if (Test-Path $runs) {
        $latest = Get-ChildItem -LiteralPath $runs -Directory -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($latest) {
            $runDest = Join-Path $Stage 'latest-run-summary'
            New-Item -ItemType Directory -Path $runDest -Force | Out-Null
            foreach ($name in @('run_config.json', 'summary.md')) {
                $source = Join-Path $latest.FullName $name
                if (Test-Path $source) { Write-RedactedFile $source (Join-Path $runDest $name) }
            }
        }
    }

    try {
        $events = Get-WinEvent -FilterHashtable @{ LogName = 'Application'; StartTime = (Get-Date).AddDays(-7) } -MaxEvents 300 -ErrorAction Stop |
            Where-Object {
                $_.ProviderName -match 'Application Error|Windows Error Reporting|\.NET Runtime|WebView2' -or
                $_.Message -match 'ChemTsCorr\.exe|WebView2'
            } | Select-Object -First 50 | ForEach-Object {
                [pscustomobject]@{
                    time_created = $_.TimeCreated
                    provider = $_.ProviderName
                    event_id = $_.Id
                    level = $_.LevelDisplayName
                    message = Redact-Text $_.Message
                }
            }
        $events | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $Stage 'application-events.json') -Encoding UTF8
    } catch {
        "WARNING: event log collection failed: $(Redact-Text $_.Exception.Message)" |
            Set-Content -LiteralPath (Join-Path $Stage 'application-events-warning.txt') -Encoding UTF8
    }

    @'
This diagnostic package intentionally excludes:
- uploads\* and original production data
- complete analysis tables
- API keys, Authorization values, tokens, passwords, and LLM secrets
- logs from other applications

Included run files are allowlisted to run_config.json and summary.md and are redacted.
'@ | Set-Content -LiteralPath (Join-Path $Stage 'COLLECTION_POLICY.txt') -Encoding UTF8

    Write-Host 'Files to be collected:'
    Get-ChildItem -LiteralPath $Stage -Recurse -File | ForEach-Object {
        Write-Host (Redact-Text $_.FullName)
    }
    if (Test-Path $ZipPath) { Remove-Item -LiteralPath $ZipPath -Force }
    Compress-Archive -Path (Join-Path $Stage '*') -DestinationPath $ZipPath -CompressionLevel Optimal
    Write-Host "Diagnostic package: $ZipPath"
    Write-Host "SHA-256: $((Get-FileHash -LiteralPath $ZipPath -Algorithm SHA256).Hash)"
} finally {
    if (Test-Path $Stage) { Remove-Item -LiteralPath $Stage -Recurse -Force }
}
