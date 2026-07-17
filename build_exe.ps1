[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error 'Python 3.10 or later was not found on PATH.'
    exit 1
}
& python --version
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& python -c "import PyInstaller, webview, sklearn, statsmodels, matplotlib, shap, xgboost"
if ($LASTEXITCODE -ne 0) {
    Write-Error 'Packaging dependencies are missing. Run: python -m pip install -e .[full,xgb] pyinstaller'
    exit $LASTEXITCODE
}

Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force dist\ChemTsCorr -ErrorAction SilentlyContinue
& python -m PyInstaller --noconfirm --clean ChemTsCorr.spec
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$Release = Join-Path $ProjectRoot 'dist\ChemTsCorr'
if (-not (Test-Path (Join-Path $Release 'ChemTsCorr.exe'))) {
    Write-Error "Build did not produce $Release\ChemTsCorr.exe"
    exit 1
}
Write-Host "Build succeeded. Release directory: $Release"
