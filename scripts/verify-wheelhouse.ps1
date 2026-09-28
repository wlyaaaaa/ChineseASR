param(
  [string]$RuntimeRoot = 'E:\Projects\Tools\ChineseASR',
  [string]$Bundle = 'E:\Projects\Tools\ChineseASR\offline\current',
  [string]$HostPython = '',
  [switch]$Seal
)

$ErrorActionPreference = 'Stop'
$Python = if ($HostPython) { $HostPython } else { Join-Path $RuntimeRoot '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
  throw "Python missing: $Python. Supply -HostPython on a restored machine."
}
$Action = if ($Seal) { 'seal' } else { 'verify' }
& $Python -B (Join-Path $PSScriptRoot 'offline_bundle.py') $Action --bundle $Bundle
if ($LASTEXITCODE -ne 0) { throw "Offline bundle $Action failed: $Bundle" }
