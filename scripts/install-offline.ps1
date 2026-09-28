param(
  [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$RuntimeRoot = 'E:\Projects\Tools\ChineseASR',
  [string]$Bundle = 'E:\Projects\Tools\ChineseASR\offline\current',
  [string]$HostPython = '',
  [Parameter(Mandatory = $true)][string]$Venv,
  [switch]$RestoreModels,
  [switch]$VerifyLinux,
  [string]$LinuxVenv = '',
  [string]$Distro = 'Ubuntu'
)

$ErrorActionPreference = 'Stop'
if (-not $HostPython) { $HostPython = Join-Path $RuntimeRoot '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $HostPython -PathType Leaf)) {
  throw 'Python 3.11 host missing. Supply -HostPython from the restored E: runtime.'
}
$Versions = Get-Content -LiteralPath (Join-Path $Bundle 'manifests\versions.json') -Raw | ConvertFrom-Json
$ActualPython = (& $HostPython --version | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $ActualPython -ne [string]$Versions.windows_python) {
  throw "Python version mismatch: backup expects $($Versions.windows_python), host is $ActualPython"
}
$Target = [IO.Path]::GetFullPath($Venv)
if ([IO.Path]::GetPathRoot($Target) -notmatch '^[Ee]:\\$') {
  throw 'Offline smoke virtual environments must be on E:.'
}
if (Test-Path -LiteralPath $Target) { throw "Smoke venv already exists: $Target" }
& (Join-Path $PSScriptRoot 'verify-wheelhouse.ps1') -RuntimeRoot $RuntimeRoot -Bundle $Bundle -HostPython $HostPython
if ($LASTEXITCODE -ne 0) { throw 'Bundle verification failed.' }
if ($RestoreModels) {
  & $HostPython -B (Join-Path $PSScriptRoot 'offline_bundle.py') restore-models `
    --runtime-root $RuntimeRoot --bundle $Bundle
  if ($LASTEXITCODE -ne 0) { throw 'Restoring model files from bundle failed.' }
}

& $HostPython -m venv $Target
if ($LASTEXITCODE -ne 0) { throw 'Cannot create Windows smoke venv.' }
$Python = Join-Path $Target 'Scripts\python.exe'
$WindowsWheelhouse = Join-Path $Bundle 'wheelhouse\windows'
$WindowsLock = Join-Path $Bundle 'manifests\windows-requirements.txt'
& $Python -m pip install --no-index --only-binary=:all: --find-links $WindowsWheelhouse -r $WindowsLock
if ($LASTEXITCODE -ne 0) { throw 'Disconnected Windows dependency install failed.' }
& $Python -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Disconnected Windows pip check failed.' }
$PreviousPythonPath = $env:PYTHONPATH
try {
  $env:PYTHONPATH = Join-Path $ProjectRoot 'src'
  & $Python -B -m zh_asr doctor
  if ($LASTEXITCODE -ne 0) { throw 'Disconnected ChineseASR doctor failed.' }
}
finally { $env:PYTHONPATH = $PreviousPythonPath }

if ($VerifyLinux) {
  if (-not $LinuxVenv) { throw '-LinuxVenv is required with -VerifyLinux.' }
  $ActualLinuxPython = (& wsl.exe -d $Distro -- python3 --version | Out-String).Trim()
  if ($LASTEXITCODE -ne 0 -or $ActualLinuxPython -ne [string]$Versions.linux_python) {
    throw "WSL Python version mismatch: backup expects $($Versions.linux_python), host is $ActualLinuxPython"
  }
  if ($LinuxVenv -notmatch '^/tmp/chineseasr-offline-[a-f0-9]+$') {
    throw 'Linux smoke venv must be a new /tmp/chineseasr-offline-<id> path in the E:-backed Ubuntu distribution.'
  }
  & wsl.exe -d $Distro -- test ! -e $LinuxVenv
  if ($LASTEXITCODE -ne 0) { throw "Linux smoke venv already exists: $LinuxVenv" }
  function Quote-Bash([string]$Value) { return "'" + $Value.Replace("'", "'`"`'`"`'") + "'" }
  function Convert-ToWslPath([string]$Path) {
    $Full = [IO.Path]::GetFullPath($Path)
    if ($Full -notmatch '^([A-Za-z]):\\(.*)$') { throw "Expected local drive path: $Full" }
    return '/mnt/' + $Matches[1].ToLowerInvariant() + '/' + $Matches[2].Replace('\', '/')
  }
  $LinuxVenvWsl = Quote-Bash $LinuxVenv
  $LinuxWheelsWsl = Quote-Bash (Convert-ToWslPath (Join-Path $Bundle 'wheelhouse\linux'))
  $LinuxLockWsl = Quote-Bash (Convert-ToWslPath (Join-Path $Bundle 'manifests\linux-requirements.txt'))
  $ArchiveWsl = Quote-Bash (Convert-ToWslPath (Join-Path $Bundle 'source\FireRedASR2S.tar'))
  $LinuxSourceWsl = Quote-Bash ($LinuxVenv + '-source')
  $Command = "set -e; python3 -m venv --copies $LinuxVenvWsl; $LinuxVenvWsl/bin/python -m pip install --no-index --only-binary=:all: --find-links $LinuxWheelsWsl -r $LinuxLockWsl; $LinuxVenvWsl/bin/python -m pip check; mkdir -p $LinuxSourceWsl; tar -xf $ArchiveWsl -C $LinuxSourceWsl; PYTHONPATH=$LinuxSourceWsl $LinuxVenvWsl/bin/python -c 'import torch,transformers,kaldi_native_fbank,fireredasr2s; print(1)'"
  & wsl.exe -d $Distro -- bash -lc $Command
  if ($LASTEXITCODE -ne 0) { throw 'Disconnected FireRed WSL smoke failed.' }
}
Write-Host 'Offline Windows dependency install, pip check and ChineseASR doctor passed.'
