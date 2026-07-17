[CmdletBinding()]
param(
    [string]$ExePath = (Join-Path $PSScriptRoot 'dist\ChemTsCorr\ChemTsCorr.exe')
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path $ExePath)) { throw "EXE not found: $ExePath" }

$process = Start-Process -FilePath $ExePath -ArgumentList '--desktop-service', '--host', '127.0.0.1', '--port', '8765' -PassThru
try {
    $deadline = (Get-Date).AddSeconds(30)
    do {
        $response = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8765/' -TimeoutSec 1 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) { break }
        Start-Sleep -Milliseconds 250
    } while ((Get-Date) -lt $deadline)
    if ($response.StatusCode -ne 200) { throw 'The packaged local homepage did not become available.' }
    Write-Host 'Packaged EXE started and local homepage returned HTTP 200.'
} finally {
    if (-not $process.HasExited) { Stop-Process -Id $process.Id -Force }
    Start-Sleep -Seconds 1
}

if (Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue) {
    throw 'Port 8765 is still in use after the EXE was closed.'
}
Write-Host 'Packaged EXE exited and port 8765 was released.'
