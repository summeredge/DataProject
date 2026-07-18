[CmdletBinding()]
param(
    [string]$Version = '',
    [ValidateSet('candidate', 'release')]
    [string]$Channel = 'candidate',
    [string]$PythonPath = 'python',
    [string]$TargetPcAcceptanceRecord = ''
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot
$PythonExe = (Get-Command $PythonPath -ErrorAction Stop).Source
$env:PATH = "$(Split-Path -Parent $PythonExe);$env:PATH"

function Invoke-RequiredStep([string]$Name, [scriptblock]$Action) {
    Write-Host "== $Name =="
    try {
        $global:LASTEXITCODE = 0
        & $Action
        if ($LASTEXITCODE -ne 0) { throw "$Name exited with code $LASTEXITCODE" }
        $Script:Tests[$Name] = 'Passed'
        return $true
    } catch {
        $Script:Tests[$Name] = "Failed: $($_.Exception.Message)"
        Write-Warning $Script:Tests[$Name]
        $Script:RequiredFailure = $true
        return $false
    }
}

function Get-RelativePath([string]$Base, [string]$Path) {
    $baseUri = New-Object Uri(($Base.TrimEnd('\') + '\'))
    $pathUri = New-Object Uri($Path)
    return [Uri]::UnescapeDataString($baseUri.MakeRelativeUri($pathUri).ToString()).Replace('/', '\')
}

$Script:RequiredFailure = $false
$Script:Tests = [ordered]@{
    pytest = 'Not Tested'
    static_packaging_tests = 'Not Tested'
    ruff = 'Not Tested'
    build_exe = 'Not Tested'
    smoke_exe = 'Not Tested'
    target_pc = 'Not Tested'
    defender_directory = 'Not Tested'
    defender_zip = 'Not Tested'
}

if (-not $Version) {
    $Version = (& $PythonExe -c "from chem_ts_corr import __version__; print(__version__)").Trim()
    if ($LASTEXITCODE -ne 0 -or -not $Version) { throw 'Unable to determine product version.' }
}
if ($Channel -eq 'candidate' -and $Version -notmatch '-(rc\d*|test|unverified)') {
    $Version = "$Version-unverified"
}
if ($Channel -eq 'release' -and $Version -match '-(rc\d*|test|unverified)') {
    throw 'Formal release versions cannot contain rc, test, or unverified.'
}

[void](Invoke-RequiredStep 'static_packaging_tests' { & $PythonExe -m pytest tests/test_exe_packaging_static.py tests/test_release_packaging.py -q })
[void](Invoke-RequiredStep 'pytest' { & $PythonExe -m pytest })
[void](Invoke-RequiredStep 'ruff' { & $PythonExe -m ruff check . })
[void](Invoke-RequiredStep 'build_exe' { & (Join-Path $ProjectRoot 'build_exe.ps1') })
[void](Invoke-RequiredStep 'smoke_exe' { & (Join-Path $ProjectRoot 'smoke_exe.ps1') })

if ($TargetPcAcceptanceRecord) {
    if (-not (Test-Path -LiteralPath $TargetPcAcceptanceRecord -PathType Leaf)) {
        throw "Target PC acceptance record not found: $TargetPcAcceptanceRecord"
    }
    $Script:Tests.target_pc = "Recorded: $([IO.Path]::GetFileName($TargetPcAcceptanceRecord))"
} elseif ($Channel -eq 'release') {
    $Script:RequiredFailure = $true
    $Script:Tests.target_pc = 'Failed: formal release requires -TargetPcAcceptanceRecord'
}

if ($Script:RequiredFailure -and $Channel -eq 'release') {
    throw 'Formal release gate failed. No release package was generated.'
}
if ($Script:RequiredFailure -and $Version -notmatch '-unverified') {
    $Version = "$Version-unverified"
}

$ReleaseOutput = Join-Path $ProjectRoot 'release-output'
$PackageName = "ChemTsCorr-$Version-windows-x64"
$PackageRoot = Join-Path $ReleaseOutput $PackageName
$AppDestination = Join-Path $PackageRoot 'ChemTsCorr'
$TestDataDestination = Join-Path $PackageRoot 'test-data'
$ZipPath = Join-Path $ReleaseOutput "$PackageName.zip"
$ZipHashPath = Join-Path $ReleaseOutput "$PackageName.zip.sha256"
$FinalManifestPath = Join-Path $ReleaseOutput 'release_manifest.final.json'

if (Test-Path $PackageRoot) { Remove-Item -LiteralPath $PackageRoot -Recurse -Force }
New-Item -ItemType Directory -Path $PackageRoot -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $ProjectRoot 'dist\ChemTsCorr') -Destination $AppDestination -Recurse

New-Item -ItemType Directory -Path $TestDataDestination -Force | Out-Null
& $PythonExe (Join-Path $ProjectRoot 'release\generate_acceptance_data.py') --output-dir $TestDataDestination
if ($LASTEXITCODE -ne 0) { throw 'Acceptance data generation failed.' }

foreach ($name in @('README_TARGET_PC.md', 'target_pc_acceptance.ps1', 'collect_diagnostics.ps1', 'acceptance_report_template.md', 'false_positive_report_template.md')) {
    Copy-Item -LiteralPath (Join-Path $ProjectRoot "release\$name") -Destination (Join-Path $PackageRoot $name)
}
if ($TargetPcAcceptanceRecord) {
    Copy-Item -LiteralPath $TargetPcAcceptanceRecord -Destination (Join-Path $PackageRoot ([IO.Path]::GetFileName($TargetPcAcceptanceRecord)))
}

$DefenderAvailable = $false
$DefenderUnavailableReason = 'Defender cmdlets unavailable or policy-controlled'
if ((Get-Command Start-MpScan -ErrorAction SilentlyContinue) -and (Get-Command Get-MpComputerStatus -ErrorAction SilentlyContinue)) {
    try {
        $DefenderStatus = Get-MpComputerStatus -ErrorAction Stop
        $DefenderAvailable = [bool]$DefenderStatus.AMServiceEnabled -and [bool]$DefenderStatus.AntivirusEnabled
        if (-not $DefenderAvailable) { $DefenderUnavailableReason = 'Defender antivirus service is disabled or inactive' }
    } catch {
        $DefenderUnavailableReason = $_.Exception.Message
    }
}
if ($DefenderAvailable) {
    try {
        Start-MpScan -ScanType CustomScan -ScanPath $PackageRoot -ErrorAction Stop
        $Script:Tests.defender_directory = 'Passed'
    } catch {
        $Script:Tests.defender_directory = "Skipped: $($_.Exception.Message)"
    }
} else {
    $Script:Tests.defender_directory = "Skipped: $DefenderUnavailableReason"
}

$TemplatePath = Join-Path $ProjectRoot 'release\release_manifest_template.json'
$Manifest = Get-Content -LiteralPath $TemplatePath -Raw | ConvertFrom-Json
$Manifest.manifest_scope = 'archive-internal'
$Manifest.product = 'ChemTsCorr'
$Manifest.version = $Version
$Manifest.build_time = (Get-Date).ToUniversalTime().ToString('o')
$Manifest.git_commit = (& git rev-parse HEAD 2>$null)
$Manifest.git_branch = (& git branch --show-current 2>$null)
$Manifest.python_version = (& $PythonExe --version 2>&1).ToString()
$Manifest.pyinstaller_version = (& $PythonExe -c "import PyInstaller; print(PyInstaller.__version__)").Trim()
$Manifest.exe_sha256 = (Get-FileHash -LiteralPath (Join-Path $AppDestination 'ChemTsCorr.exe') -Algorithm SHA256).Hash
$Manifest.archive_name = $null
$Manifest.zip_sha256 = $null
$Manifest.tests = [pscustomobject]$Script:Tests
$payloadFiles = Get-ChildItem -LiteralPath $PackageRoot -Recurse -File
$Manifest.release_size_bytes = [int64](($payloadFiles | Measure-Object Length -Sum).Sum)
$keyHashes = [ordered]@{}
foreach ($path in @(
    (Join-Path $AppDestination 'ChemTsCorr.exe'),
    (Join-Path $PackageRoot 'target_pc_acceptance.ps1'),
    (Join-Path $PackageRoot 'collect_diagnostics.ps1')
)) {
    $keyHashes[(Get-RelativePath $PackageRoot $path)] = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
}
$Manifest.key_file_sha256 = [pscustomobject]$keyHashes
$InternalManifestPath = Join-Path $PackageRoot 'release_manifest.json'
$Manifest | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $InternalManifestPath -Encoding UTF8

$sumLines = Get-ChildItem -LiteralPath $PackageRoot -Recurse -File | Sort-Object FullName | ForEach-Object {
    $relative = Get-RelativePath $PackageRoot $_.FullName
    "$((Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash)  $relative"
}
$sumLines | Set-Content -LiteralPath (Join-Path $PackageRoot 'SHA256SUMS.txt') -Encoding ASCII

if (Test-Path $ZipPath) { Remove-Item -LiteralPath $ZipPath -Force }
Compress-Archive -LiteralPath $PackageRoot -DestinationPath $ZipPath -CompressionLevel Optimal
$ZipHash = (Get-FileHash -LiteralPath $ZipPath -Algorithm SHA256).Hash
"$ZipHash  $([IO.Path]::GetFileName($ZipPath))" | Set-Content -LiteralPath $ZipHashPath -Encoding ASCII

if ($DefenderAvailable) {
    try {
        Start-MpScan -ScanType CustomScan -ScanPath $ZipPath -ErrorAction Stop
        $Script:Tests.defender_zip = 'Passed'
    } catch {
        $Script:Tests.defender_zip = "Skipped: $($_.Exception.Message)"
    }
} else {
    $Script:Tests.defender_zip = "Skipped: $DefenderUnavailableReason"
}

$FinalManifest = Get-Content -LiteralPath $InternalManifestPath -Raw | ConvertFrom-Json
$FinalManifest.manifest_scope = "external-release"
$FinalManifest.archive_name = [IO.Path]::GetFileName($ZipPath)
$FinalManifest.zip_sha256 = $ZipHash
$FinalManifest.tests = [pscustomobject]$Script:Tests
$FinalManifest | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $FinalManifestPath -Encoding UTF8

Write-Host "Expanded release: $PackageRoot"
Write-Host "Archive: $ZipPath"
Write-Host "Archive size: $((Get-Item -LiteralPath $ZipPath).Length) bytes"
Write-Host "Archive SHA-256: $ZipHash"
if ($Script:RequiredFailure) {
    Write-Warning 'Candidate package contains failed checks and is marked unverified.'
    exit 1
}
