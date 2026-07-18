[CmdletBinding()]
param(
    [string]$ExePath,
    [string]$TestDataDir,
    [string]$ReportPath,
    [switch]$Interactive,
    [switch]$SkipLargeData,
    [switch]$SkipDefenderCheck,
    [int]$SmallDataTimeoutMinutes = 5,
    [int]$LargeDataTimeoutMinutes = 30,
    [int]$RestartCount = 10
)

$ErrorActionPreference = 'Stop'
$Script:Results = New-Object System.Collections.ArrayList
$Script:CriticalFailure = $false
$Timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
if (-not $ExePath) { $ExePath = Join-Path $PSScriptRoot 'ChemTsCorr\ChemTsCorr.exe' }
if (-not $TestDataDir) { $TestDataDir = Join-Path $PSScriptRoot 'test-data' }
if (-not $ReportPath) {
    $reportBase = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { $env:TEMP }
    $ReportPath = Join-Path $reportBase 'ChemTsCorr\acceptance-results'
}
$ExePath = [IO.Path]::GetFullPath($ExePath)
$TestDataDir = [IO.Path]::GetFullPath($TestDataDir)
$ReportPath = [IO.Path]::GetFullPath($ReportPath)
New-Item -ItemType Directory -Path $ReportPath -Force | Out-Null
$LogPath = Join-Path $ReportPath "target-pc-acceptance-$Timestamp.log"
$JsonPath = Join-Path $ReportPath "target-pc-acceptance-$Timestamp.json"
$MarkdownPath = Join-Path $ReportPath "target-pc-acceptance-$Timestamp.md"

function Write-Log([string]$Message) {
    $line = "{0} {1}" -f (Get-Date -Format 'yyyy-MM-ddTHH:mm:ss.fffK'), $Message
    Add-Content -LiteralPath $LogPath -Value $line -Encoding UTF8
    Write-Host $line
}

function Sanitize-Text([string]$Text) {
    if ($null -eq $Text) { return $null }
    $result = $Text
    if ($env:USERPROFILE) { $result = $result.Replace($env:USERPROFILE, '%USERPROFILE%') }
    if ($env:USERNAME) { $result = $result.Replace($env:USERNAME, '[REDACTED]') }
    if ($env:COMPUTERNAME) { $result = $result.Replace($env:COMPUTERNAME, '[REDACTED]') }
    return $result
}

function Add-Result(
    [string]$Name,
    [string]$Status,
    [datetime]$Started,
    [string]$ErrorMessage = '',
    [object[]]$Pids = @(),
    [object[]]$Ports = @(),
    [object[]]$Outputs = @(),
    [string]$Notes = '',
    [bool]$Critical = $true
) {
    $ended = Get-Date
    $item = [ordered]@{
        name = $Name
        started_at = $Started.ToString('o')
        ended_at = $ended.ToString('o')
        duration_seconds = [math]::Round(($ended - $Started).TotalSeconds, 3)
        status = $Status
        error = Sanitize-Text $ErrorMessage
        pids = @($Pids)
        ports = @($Ports)
        output_files = @($Outputs | ForEach-Object { Sanitize-Text ([string]$_) })
        notes = Sanitize-Text $Notes
        critical = $Critical
    }
    [void]$Script:Results.Add([pscustomobject]$item)
    if ($Critical -and $Status -eq 'Fail') { $Script:CriticalFailure = $true }
    Write-Log "$Status - $Name$(if ($ErrorMessage) { ": $ErrorMessage" })"
}

function Invoke-Check([string]$Name, [scriptblock]$Action, [bool]$Critical = $true) {
    $started = Get-Date
    try {
        $evidence = & $Action
        if ($null -eq $evidence) { $evidence = [pscustomobject]@{} }
        Add-Result -Name $Name -Status 'Pass' -Started $started -Pids $evidence.pids `
            -Ports $evidence.ports -Outputs $evidence.outputs -Notes $evidence.notes -Critical $Critical
    } catch {
        Add-Result -Name $Name -Status 'Fail' -Started $started `
            -ErrorMessage $_.Exception.Message -Critical $Critical
    }
}

function Add-Skipped([string]$Name, [string]$Reason, [bool]$Critical = $false) {
    Add-Result -Name $Name -Status 'Skipped' -Started (Get-Date) -Notes $Reason -Critical $Critical
}

function Get-WebView2Info {
    $keys = @(
        'HKLM:\SOFTWARE\Microsoft\EdgeUpdate\Clients\{F1E7E6A3-2D3C-4B5A-B5E4-4F5C2D1A0A1B}',
        'HKLM:\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F1E7E6A3-2D3C-4B5A-B5E4-4F5C2D1A0A1B}',
        'HKCU:\Software\Microsoft\EdgeUpdate\Clients'
    )
    $versions = @()
    foreach ($key in $keys) {
        if (Test-Path $key) {
            $versions += Get-ItemProperty -Path $key -ErrorAction SilentlyContinue |
                ForEach-Object { $_.pv } | Where-Object { $_ }
        }
    }
    $installPaths = @(
        "${env:ProgramFiles(x86)}\Microsoft\EdgeWebView\Application",
        "$env:ProgramFiles\Microsoft\EdgeWebView\Application",
        "$env:LOCALAPPDATA\Microsoft\EdgeWebView\Application"
    ) | Where-Object { $_ -and (Test-Path $_) }
    [pscustomobject]@{ registry_versions = @($versions); install_paths = @($installPaths) }
}

function Get-VcRuntimeInfo {
    $keys = @(
        'HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64',
        'HKLM:\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64'
    )
    $found = @()
    foreach ($key in $keys) {
        if (Test-Path $key) {
            $value = Get-ItemProperty -Path $key -ErrorAction SilentlyContinue
            $found += [pscustomobject]@{ key = $key; installed = $value.Installed; version = $value.Version }
        }
    }
    [pscustomobject]@{
        registrations = @($found)
        watched_dlls = @('VCRUNTIME140.dll', 'VCRUNTIME140_1.dll', 'MSVCP140.dll', 'concrt140.dll', 'api-ms-win-*.dll')
    }
}

function Get-DefenderInfo {
    if (-not (Get-Command Get-MpComputerStatus -ErrorAction SilentlyContinue)) {
        return [pscustomobject]@{ available = $false; reason = 'Get-MpComputerStatus unavailable' }
    }
    try {
        $status = Get-MpComputerStatus -ErrorAction Stop
        return [pscustomobject]@{
            available = $true
            antivirus_enabled = $status.AntivirusEnabled
            engine_version = $status.AMEngineVersion
            signature_version = $status.AntivirusSignatureVersion
            signature_updated = $status.AntivirusSignatureLastUpdated
        }
    } catch {
        return [pscustomobject]@{ available = $false; reason = $_.Exception.Message }
    }
}

function Get-SystemInfo {
    $os = Get-CimInstance Win32_OperatingSystem
    $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
    $drive = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='$([IO.Path]::GetPathRoot($ExePath).TrimEnd('\'))'"
    [ordered]@{
        windows_caption = $os.Caption
        windows_version = $os.Version
        windows_build = $os.BuildNumber
        architecture = $os.OSArchitecture
        cpu = $cpu.Name
        logical_cores = $cpu.NumberOfLogicalProcessors
        memory_total_bytes = [int64]$os.TotalVisibleMemorySize * 1KB
        memory_free_bytes = [int64]$os.FreePhysicalMemory * 1KB
        culture = [Globalization.CultureInfo]::CurrentCulture.Name
        ui_culture = [Globalization.CultureInfo]::CurrentUICulture.Name
        powershell_version = $PSVersionTable.PSVersion.ToString()
        current_user = '[REDACTED]'
        current_path = Sanitize-Text (Get-Location).Path
        disk_free_bytes = if ($drive) { [int64]$drive.FreeSpace } else { $null }
        webview2 = Get-WebView2Info
        vc_runtime = Get-VcRuntimeInfo
        defender = Get-DefenderInfo
    }
}

function Wait-ForUrl([string]$Url, [int]$TimeoutSeconds = 30) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 2
            if ($response.StatusCode -eq 200) { return }
        } catch { }
        Start-Sleep -Milliseconds 250
    } while ((Get-Date) -lt $deadline)
    throw "Timed out waiting for $Url"
}

function Wait-ForExit([int]$ProcessId, [int]$TimeoutSeconds = 15) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        if (-not (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) { return }
        Start-Sleep -Milliseconds 250
    } while ((Get-Date) -lt $deadline)
    throw "Process $ProcessId did not exit within $TimeoutSeconds seconds"
}

function Wait-ForService([int]$ParentPid) {
    $deadline = (Get-Date).AddSeconds(30)
    do {
        $child = Get-CimInstance Win32_Process -Filter "ParentProcessId=$ParentPid" -ErrorAction SilentlyContinue |
            Where-Object { $_.CommandLine -match '--desktop-service' } | Select-Object -First 1
        if ($child -and $child.CommandLine -match '--port\s+"?(\d+)"?') {
            return [pscustomobject]@{
                pid = [int]$child.ProcessId
                port = [int]$Matches[1]
                command_line = $child.CommandLine
                creation_date = $child.CreationDate
            }
        }
        if (-not (Get-Process -Id $ParentPid -ErrorAction SilentlyContinue)) {
            throw 'Desktop process exited before starting its service'
        }
        Start-Sleep -Milliseconds 250
    } while ((Get-Date) -lt $deadline)
    throw 'Desktop service was not found within 30 seconds'
}

function Start-Desktop([string]$Path) {
    $desktop = Start-Process -FilePath $Path -PassThru
    try {
        $service = Wait-ForService $desktop.Id
        Wait-ForUrl "http://127.0.0.1:$($service.port)/"
        $deadline = (Get-Date).AddSeconds(30)
        do {
            $desktop.Refresh()
            if ($desktop.MainWindowHandle -ne [IntPtr]::Zero) { break }
            Start-Sleep -Milliseconds 250
        } while ((Get-Date) -lt $deadline)
        if ($desktop.MainWindowHandle -eq [IntPtr]::Zero) { throw 'Desktop window was not created' }
        $listeners = @(Get-NetTCPConnection -OwningProcess $service.pid -State Listen -ErrorAction SilentlyContinue)
        if (-not $listeners) { throw "No listening port found for service PID $($service.pid)" }
        if ($listeners.LocalAddress | Where-Object { $_ -notin @('127.0.0.1', '::1') }) {
            throw 'Service is listening on a non-loopback address'
        }
        return [pscustomobject]@{ desktop = $desktop; service = $service }
    } catch {
        if (-not $desktop.HasExited) { & taskkill /PID $desktop.Id /T /F 2>$null | Out-Null }
        throw
    }
}

function Stop-Desktop($Instance) {
    $desktop = $Instance.desktop
    if (-not $desktop.HasExited) {
        if (-not $desktop.CloseMainWindow()) { throw 'Desktop window rejected normal close' }
        Wait-ForExit $desktop.Id
    }
    Wait-ForExit $Instance.service.pid
    if (Get-NetTCPConnection -LocalPort $Instance.service.port -State Listen -ErrorAction SilentlyContinue) {
        throw "Port $($Instance.service.port) remained open after shutdown"
    }
}

function Stop-DesktopFallback($Instance) {
    if ($Instance -and (Get-Process -Id $Instance.desktop.Id -ErrorAction SilentlyContinue)) {
        & taskkill /PID $Instance.desktop.Id /T /F 2>$null | Out-Null
    }
    if ($Instance -and (Get-Process -Id $Instance.service.pid -ErrorAction SilentlyContinue)) {
        & taskkill /PID $Instance.service.pid /T /F 2>$null | Out-Null
    }
}

function Invoke-MultipartJson([string]$Url, [hashtable]$Fields, [string]$FilePath = '') {
    Add-Type -AssemblyName System.Net.Http
    $client = New-Object System.Net.Http.HttpClient
    $content = New-Object System.Net.Http.MultipartFormDataContent
    try {
        foreach ($entry in $Fields.GetEnumerator()) {
            $field = New-Object System.Net.Http.StringContent([string]$entry.Value)
            $content.Add($field, [string]$entry.Key)
        }
        if ($FilePath) {
            $bytes = [IO.File]::ReadAllBytes($FilePath)
            $file = New-Object System.Net.Http.ByteArrayContent -ArgumentList @(,$bytes)
            $content.Add($file, 'file', [IO.Path]::GetFileName($FilePath))
        }
        $response = $client.PostAsync($Url, $content).Result
        $body = $response.Content.ReadAsStringAsync().Result
        if (-not $response.IsSuccessStatusCode) { throw "HTTP $([int]$response.StatusCode): $body" }
        return $body | ConvertFrom-Json
    } finally {
        $content.Dispose()
        $client.Dispose()
    }
}

function Invoke-AnalysisFile($Instance, [string]$DataPath, [int]$TimeoutMinutes) {
    $baseUrl = "http://127.0.0.1:$($Instance.service.port)"
    $upload = Invoke-MultipartJson "$baseUrl/api/upload" @{} $DataPath
    if (-not $upload.file_id) { throw 'Upload did not return file_id' }
    $columns = Invoke-RestMethod -Uri "$baseUrl/api/columns?file_id=$($upload.file_id)&encoding=auto"
    $chineseTarget = -join (0x76EE, 0x6807, 0x53D8, 0x91CF | ForEach-Object { [char]$_ })
    $chineseTime = -join (0x65F6, 0x95F4 | ForEach-Object { [char]$_ })
    $isChinese = $columns.columns -contains $chineseTarget
    $timeColumn = if ($isChinese) { $chineseTime } else { 'timestamp' }
    $target = if ($isChinese) { $chineseTarget } else { 'target' }
    if ($columns.columns -notcontains $timeColumn -or $columns.columns -notcontains $target) {
        throw "Required columns were not recognized in $DataPath"
    }
    $started = Get-Date
    $cpuStart = (Get-Process -Id $Instance.service.pid).TotalProcessorTime.TotalSeconds
    $peakWorkingSet = 0L
    $analysis = Invoke-MultipartJson "$baseUrl/api/analyze" @{
        file_id = $upload.file_id; encoding = 'auto'; time_column = $timeColumn; target = $target
        max_lag = '12'; top_k = '30'; min_valid_ratio = '0.7'
    }
    if (-not $analysis.task_id) { throw 'Analysis did not return task_id' }
    $deadline = (Get-Date).AddMinutes($TimeoutMinutes)
    do {
        Start-Sleep -Seconds 1
        $process = Get-Process -Id $Instance.service.pid -ErrorAction Stop
        $peakWorkingSet = [math]::Max($peakWorkingSet, [int64]$process.WorkingSet64)
        $status = Invoke-RestMethod -Uri "$baseUrl/api/status?task_id=$($analysis.task_id)"
    } while ($status.status -eq 'running' -and (Get-Date) -lt $deadline)
    if ($status.status -eq 'running') { throw "Analysis timed out after $TimeoutMinutes minutes" }
    if ($status.status -ne 'done') { throw "Analysis failed: $($status.error)" }
    $result = Invoke-RestMethod -Uri "$baseUrl/api/result?task_id=$($analysis.task_id)"
    $download = $result.downloads | Select-Object -First 1
    if (-not $download.url) { throw 'No downloadable result was returned' }
    $downloadPath = Join-Path $ReportPath ("download-{0}-{1}" -f $analysis.task_id, $download.name)
    Invoke-WebRequest -UseBasicParsing -Uri "$baseUrl$($download.url)" -OutFile $downloadPath
    if ((Get-Item $downloadPath).Length -le 0) { throw 'Downloaded result is empty' }
    $elapsed = (Get-Date) - $started
    $cpuEnd = (Get-Process -Id $Instance.service.pid).TotalProcessorTime.TotalSeconds
    $cores = [Environment]::ProcessorCount
    $averageCpu = if ($elapsed.TotalSeconds -gt 0) {
        [math]::Round(100 * ($cpuEnd - $cpuStart) / ($elapsed.TotalSeconds * $cores), 2)
    } else { 0 }
    [pscustomobject]@{
        output = $downloadPath
        seconds = [math]::Round($elapsed.TotalSeconds, 3)
        peak_working_set_bytes = $peakWorkingSet
        average_cpu_percent = $averageCpu
        run_id = $result.run_id
    }
}

function Get-DirectorySnapshot([string]$Path) {
    @((Get-ChildItem -LiteralPath $Path -Recurse -File -Force).FullName |
        ForEach-Object { $_.Substring($Path.Length).TrimStart('\') } | Sort-Object)
}

function Copy-Release([string]$Source, [string]$Destination) {
    if (Test-Path $Destination) { Remove-Item -LiteralPath $Destination -Recurse -Force }
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    Copy-Item -Path (Join-Path $Source '*') -Destination $Destination -Recurse -Force
}

if (-not (Test-Path -LiteralPath $ExePath -PathType Leaf)) { throw "EXE not found: $ExePath" }
if (-not (Test-Path -LiteralPath $TestDataDir -PathType Container)) { throw "Test data not found: $TestDataDir" }
$ReleaseDir = Split-Path -Parent $ExePath
$UserDataRoot = Join-Path $env:LOCALAPPDATA 'ChemTsCorr'
$ManifestPath = Join-Path $PSScriptRoot 'release_manifest.json'
$ReleaseVersion = if (Test-Path $ManifestPath) {
    (Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json).version
} else { '' }
$SystemInfo = Get-SystemInfo
$ExeItem = Get-Item -LiteralPath $ExePath
$ExeInfo = [ordered]@{
    path = Sanitize-Text $ExePath
    file_version = $ExeItem.VersionInfo.FileVersion
    product_version = $ExeItem.VersionInfo.ProductVersion
    sha256 = (Get-FileHash -LiteralPath $ExePath -Algorithm SHA256).Hash
    release_version = $ReleaseVersion
}

Write-Log "Starting target PC acceptance for $($ExeInfo.path)"

Invoke-Check 'User data directories are writable and outside the release' {
    $releaseFull = [IO.Path]::GetFullPath($ReleaseDir).TrimEnd('\') + '\'
    $userFull = [IO.Path]::GetFullPath($UserDataRoot).TrimEnd('\') + '\'
    if ($userFull.StartsWith($releaseFull, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'User data root is inside the release directory'
    }
    foreach ($name in @('logs', 'uploads', 'web_runs')) {
        $directory = Join-Path $UserDataRoot $name
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
        $probePath = Join-Path $directory ".acceptance-write-$Timestamp.tmp"
        Set-Content -LiteralPath $probePath -Value 'write probe' -Encoding ASCII
        Remove-Item -LiteralPath $probePath -Force
    }
    [pscustomobject]@{ outputs = @($UserDataRoot); notes = 'logs, uploads, and web_runs are writable.' }
}

$driveRootAvailable = $false
$probe = 'C:\ChemTsCorrAcceptanceWriteProbe'
try {
    New-Item -ItemType Directory -Path $probe -Force | Out-Null
    Remove-Item -LiteralPath $probe -Force
    $driveRootAvailable = $true
} catch {
    Write-Log 'C:\ is not writable; path tests will use TEMP.'
}
$base = if ($driveRootAvailable) { 'C:\' } else { $env:TEMP }
$chinesePath = -join (0x5316, 0x5DE5, 0x5206, 0x6790, 0x6D4B, 0x8BD5 | ForEach-Object { [char]$_ })
$chineseSpacePath = (-join (0x5316, 0x5DE5 | ForEach-Object { [char]$_ })) + ' ' +
    (-join (0x5206, 0x6790 | ForEach-Object { [char]$_ })) + ' ' +
    (-join (0x6D4B, 0x8BD5 | ForEach-Object { [char]$_ }))
$pathNames = @(
    'ChemTsCorrTest\ChemTsCorr', 'Program Test\ChemTsCorr',
    "$chinesePath\ChemTsCorr", "$chineseSpacePath\ChemTsCorr"
)
$smallCsv = Join-Path $TestDataDir 'acceptance_small.csv'
$OriginalReleaseSnapshot = Get-DirectorySnapshot $ReleaseDir

foreach ($relative in $pathNames) {
    $destination = Join-Path $base $relative
    Invoke-Check "Path compatibility: $relative" {
        Copy-Release $ReleaseDir $destination
        $before = Get-DirectorySnapshot $destination
        $instance = $null
        try {
            $instance = Start-Desktop (Join-Path $destination 'ChemTsCorr.exe')
            $analysis = Invoke-AnalysisFile $instance $smallCsv $SmallDataTimeoutMinutes
            Stop-Desktop $instance
            $after = Get-DirectorySnapshot $destination
            $written = @($after | Where-Object { $_ -notin $before })
            if ($written) { throw "Release directory was modified: $($written -join ', ')" }
            [pscustomobject]@{
                pids = @($instance.desktop.Id, $instance.service.pid)
                ports = @($instance.service.port)
                outputs = @($analysis.output)
                notes = "HTTP 200, upload, columns, analysis, download, normal close; $($analysis.seconds)s"
            }
        } finally {
            Stop-DesktopFallback $instance
            if (Test-Path $destination) { Remove-Item -LiteralPath $destination -Recurse -Force }
        }
    }
}

Invoke-Check 'CSV TXT TSV XLSX and Chinese-column upload/analyze/download' {
    $instance = $null
    $outputs = @()
    try {
        $instance = Start-Desktop $ExePath
        $files = @(
            'acceptance_small.csv', 'acceptance_small.txt', 'acceptance_small.tsv',
            'acceptance_small.xlsx', 'acceptance_chinese_columns.csv', 'acceptance_chinese_columns.xlsx'
        )
        foreach ($file in $files) {
            $result = Invoke-AnalysisFile $instance (Join-Path $TestDataDir $file) $SmallDataTimeoutMinutes
            $outputs += $result.output
        }
        Stop-Desktop $instance
        [pscustomobject]@{
            pids = @($instance.desktop.Id, $instance.service.pid)
            ports = @($instance.service.port)
            outputs = $outputs
            notes = 'XLS and XLSM require separately supplied valid samples and remain manual.'
        }
    } finally { Stop-DesktopFallback $instance }
}

Invoke-Check 'Sequential restart 10 times' {
    $pids = @(); $ports = @()
    for ($index = 1; $index -le $RestartCount; $index++) {
        $instance = $null
        try {
            $instance = Start-Desktop $ExePath
            $pids += @($instance.desktop.Id, $instance.service.pid)
            $ports += $instance.service.port
            Stop-Desktop $instance
        } finally { Stop-DesktopFallback $instance }
    }
    [pscustomobject]@{ pids = $pids; ports = $ports; notes = "$RestartCount sequential starts completed" }
}

Invoke-Check 'Concurrent instances' {
    $first = $null; $second = $null
    try {
        $first = Start-Desktop $ExePath
        $second = Start-Desktop $ExePath
        if ($first.service.port -eq $second.service.port) { throw 'Concurrent instances reused one port' }
        Stop-Desktop $first
        Wait-ForUrl "http://127.0.0.1:$($second.service.port)/" 10
        Stop-Desktop $second
        [pscustomobject]@{
            pids = @($first.desktop.Id, $first.service.pid, $second.desktop.Id, $second.service.pid)
            ports = @($first.service.port, $second.service.port)
            notes = 'Closing the first instance did not stop the second.'
        }
    } finally { Stop-DesktopFallback $first; Stop-DesktopFallback $second }
}

Invoke-Check 'Forced process-tree shutdown and Port release' {
    $instance = $null
    try {
        $instance = Start-Desktop $ExePath
        & taskkill /PID $instance.desktop.Id /T /F 2>$null | Out-Null
        Wait-ForExit $instance.desktop.Id
        Wait-ForExit $instance.service.pid
        if (Get-NetTCPConnection -LocalPort $instance.service.port -State Listen -ErrorAction SilentlyContinue) {
            throw 'Port remained open after forced process-tree shutdown'
        }
        [pscustomobject]@{
            pids = @($instance.desktop.Id, $instance.service.pid)
            ports = @($instance.service.port)
            notes = 'taskkill /T /F cleanup and restart recovery verified'
        }
    } finally { Stop-DesktopFallback $instance }
    $retry = $null
    try { $retry = Start-Desktop $ExePath; Stop-Desktop $retry } finally { Stop-DesktopFallback $retry }
}

Invoke-Check 'Fixed smoke port conflict fails clearly' {
    $listener = New-Object Net.Sockets.TcpListener([Net.IPAddress]::Loopback, 8765)
    $listener.ExclusiveAddressUse = $true
    $process = $null
    try {
        $listener.Start()
        $process = Start-Process -FilePath $ExePath -ArgumentList '--desktop-service', '--host', '127.0.0.1', '--port', '8765' -PassThru
        Wait-ForExit $process.Id 15
        if ($process.ExitCode -eq 0) { throw 'Fixed-port service unexpectedly succeeded while port 8765 was occupied' }
        [pscustomobject]@{ pids = @($process.Id); ports = @(8765); notes = "Exit code $($process.ExitCode)" }
    } finally {
        if ($process -and -not $process.HasExited) { & taskkill /PID $process.Id /T /F 2>$null | Out-Null }
        $listener.Stop()
    }
}

if ($SkipLargeData) {
    Add-Skipped '40x45000 large-data primary screening' 'Skipped by -SkipLargeData.'
} else {
    Invoke-Check '40x45000 large-data primary screening' {
        $instance = $null
        try {
            $instance = Start-Desktop $ExePath
            $analysis = Invoke-AnalysisFile $instance (Join-Path $TestDataDir 'acceptance_large.csv') $LargeDataTimeoutMinutes
            Stop-Desktop $instance
            [pscustomobject]@{
                pids = @($instance.desktop.Id, $instance.service.pid); ports = @($instance.service.port)
                outputs = @($analysis.output)
                notes = "duration=$($analysis.seconds)s; peak_memory=$($analysis.peak_working_set_bytes); avg_cpu=$($analysis.average_cpu_percent)%"
            }
        } finally { Stop-DesktopFallback $instance }
    }
}

if ($SkipDefenderCheck) {
    Add-Skipped 'Microsoft Defender release-directory scan' 'Skipped by -SkipDefenderCheck.'
} elseif (-not (Get-Command Start-MpScan -ErrorAction SilentlyContinue)) {
    Add-Skipped 'Microsoft Defender release-directory scan' 'Start-MpScan is unavailable or blocked by policy.'
} else {
    $defenderStarted = Get-Date
    try {
        Start-MpScan -ScanType CustomScan -ScanPath $ReleaseDir -ErrorAction Stop
        Add-Result -Name 'Microsoft Defender release-directory scan' -Status 'Pass' `
            -Started $defenderStarted -Notes 'Custom scan command completed.' -Critical $false
    } catch {
        Add-Result -Name 'Microsoft Defender release-directory scan' -Status 'Skipped' `
            -Started $defenderStarted -Notes "Unavailable or policy-controlled: $($_.Exception.Message)" -Critical $false
    }
}

foreach ($manual in @(
    'No-Python target machine confirmation', 'VC++ Runtime absent machine', 'WebView2 absent/corrupt machine',
    'XLS and XLSM valid sample upload', 'Granger execution', 'SHAP execution', 'XGBoost execution',
    'Window close during upload/analysis/download', 'Task Manager kills main process without process-tree option',
    'Windows sign-out/shutdown', 'Unwritable user directory', 'Insufficient disk space', 'Locked output file'
)) {
    Add-Skipped "Manual: $manual" 'Requires a human-controlled target-machine scenario; no automatic pass was claimed.'
}

Invoke-Check 'Original release directory remained unchanged' {
    $after = Get-DirectorySnapshot $ReleaseDir
    $added = @($after | Where-Object { $_ -notin $OriginalReleaseSnapshot })
    $removed = @($OriginalReleaseSnapshot | Where-Object { $_ -notin $after })
    if ($added -or $removed) {
        throw "Release directory changed. Added: $($added -join ', '); removed: $($removed -join ', ')"
    }
    [pscustomobject]@{ notes = 'No files were added to or removed from the executable directory.' }
}

$Report = [ordered]@{
    schema_version = 1
    generated_at = (Get-Date).ToString('o')
    redacted = $true
    system = $SystemInfo
    executable = $ExeInfo
    report_files = [ordered]@{ json = Sanitize-Text $JsonPath; markdown = Sanitize-Text $MarkdownPath; log = Sanitize-Text $LogPath }
    tests = @($Script:Results)
    summary = [ordered]@{
        pass = @($Script:Results | Where-Object status -eq 'Pass').Count
        fail = @($Script:Results | Where-Object status -eq 'Fail').Count
        skipped = @($Script:Results | Where-Object status -eq 'Skipped').Count
        critical_failure = $Script:CriticalFailure
    }
}
$Report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $JsonPath -Encoding UTF8

$md = New-Object Text.StringBuilder
[void]$md.AppendLine('# ChemTsCorr target PC acceptance result')
[void]$md.AppendLine()
[void]$md.AppendLine("- Generated: $($Report.generated_at)")
[void]$md.AppendLine("- Release version: $ReleaseVersion")
[void]$md.AppendLine("- EXE SHA-256: $($ExeInfo.sha256)")
[void]$md.AppendLine("- Summary: Pass $($Report.summary.pass) / Fail $($Report.summary.fail) / Skipped $($Report.summary.skipped)")
[void]$md.AppendLine('- Redacted: user and machine names are omitted.')
[void]$md.AppendLine()
[void]$md.AppendLine('| Test | Status | PID | Port | Notes/Error |')
[void]$md.AppendLine('| --- | --- | --- | --- | --- |')
foreach ($item in $Script:Results) {
    $detail = if ($item.error) { $item.error } else { $item.notes }
    $detail = ($detail -replace '\|', '\|') -replace "`r?`n", ' '
    [void]$md.AppendLine("| $($item.name) | $($item.status) | $($item.pids -join ',') | $($item.ports -join ',') | $detail |")
}
[void]$md.AppendLine()
[void]$md.AppendLine('## Manual acceptance')
[void]$md.AppendLine()
[void]$md.AppendLine('Every `Manual:` item remains `Skipped` until a human records the real target-machine result.')
$md.ToString() | Set-Content -LiteralPath $MarkdownPath -Encoding UTF8

Write-Log "JSON report: $JsonPath"
Write-Log "Markdown report: $MarkdownPath"
if ($Script:CriticalFailure) { exit 1 }
exit 0
