param(
  [string]$Venv = '',
  [string]$OutDir = ''
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Invoke-NoProxy.ps1')
Clear-ProxyEnv

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $Venv) {
  $Venv = Join-Path $Root '.venv'
}
if (-not $OutDir) {
  $OutDir = Join-Path $Root 'offline\manifests'
}

$Python = Join-Path $Venv 'Scripts\python.exe'
if (-not (Test-Path $Python)) {
  throw "Virtual environment not found: $Venv"
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$LockFile = Join-Path $OutDir 'requirements-lock.txt'
$PythonVersionFile = Join-Path $OutDir 'python-version.txt'

$Freeze = & $Python -m pip freeze --exclude-editable
if ($LASTEXITCODE -ne 0) {
  throw 'pip freeze failed.'
}
$Freeze |
  Where-Object { $_ -and ($_ -notmatch '^-e\s+') } |
  Set-Content -Encoding UTF8 $LockFile

& $Python --version | Set-Content -Encoding UTF8 $PythonVersionFile
if ($LASTEXITCODE -ne 0) {
  throw 'python --version failed.'
}

& $Python -m pip check
if ($LASTEXITCODE -ne 0) {
  throw 'pip check failed.'
}

Write-Host "Lock file: $LockFile"
Write-Host "Python version: $PythonVersionFile"
