param(
  [string]$Venv = '',
  [string]$Wheelhouse = '',
  [string]$ManifestDir = '',
  [switch]$SkipVerify
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Invoke-NoProxy.ps1')
Clear-ProxyEnv

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $Venv) {
  $Venv = Join-Path $Root '.venv'
}
if (-not $Wheelhouse) {
  $Wheelhouse = Join-Path $Root 'offline\wheelhouse'
}
if (-not $ManifestDir) {
  $ManifestDir = Join-Path $Root 'offline\manifests'
}

$LockFile = Join-Path $ManifestDir 'requirements-lock.txt'
$ChecksumFile = Join-Path $ManifestDir 'wheelhouse.sha256'
if (-not (Test-Path $LockFile)) {
  throw "Lock file not found: $LockFile"
}

if (-not $SkipVerify) {
  & (Join-Path $PSScriptRoot 'verify-wheelhouse.ps1') -Wheelhouse $Wheelhouse -ChecksumFile $ChecksumFile
  if ($LASTEXITCODE -ne 0) {
    throw 'verify-wheelhouse.ps1 failed.'
  }
}

if (-not (Test-Path $Venv)) {
  python -m venv $Venv
}

$Python = Join-Path $Venv 'Scripts\python.exe'
if (-not (Test-Path $Python)) {
  throw "Python not found in virtual environment: $Python"
}

& $Python -m pip install --no-index --find-links $Wheelhouse -r $LockFile
if ($LASTEXITCODE -ne 0) {
  throw 'Offline pip install failed.'
}

& $Python -m pip install -e $Root --no-deps
if ($LASTEXITCODE -ne 0) {
  throw 'Editable project install failed.'
}

& $Python -m pip check
if ($LASTEXITCODE -ne 0) {
  throw 'pip check failed.'
}

& $Python -m zh_asr doctor
if ($LASTEXITCODE -ne 0) {
  throw 'zh_asr doctor failed.'
}
