param(
  [string]$RuntimeRoot = 'E:\Projects\Tools\ChineseASR',
  [Parameter(Mandatory = $true)][string]$Bundle,
  [string]$Distro = 'Ubuntu'
)

$ErrorActionPreference = 'Stop'
$Python = Join-Path $RuntimeRoot '.venv\Scripts\python.exe'
$WindowsLock = Join-Path $Bundle 'manifests\windows-requirements.txt'
$LinuxLock = Join-Path $Bundle 'manifests\linux-requirements.txt'
$VersionsPath = Join-Path $Bundle 'manifests\versions.json'
foreach ($item in @($Python, $WindowsLock, $LinuxLock, $VersionsPath)) {
  if (-not (Test-Path -LiteralPath $item -PathType Leaf)) { throw "Missing installed runtime or lock: $item" }
}
if ([IO.Path]::GetPathRoot([IO.Path]::GetFullPath($Bundle)) -notmatch '^[Ee]:\\$') {
  throw 'Offline bundle must be staged on E:.'
}
$WindowsWheelhouse = Join-Path $Bundle 'wheelhouse\windows'
$LinuxWheelhouse = Join-Path $Bundle 'wheelhouse\linux'
New-Item -ItemType Directory -Force -Path $WindowsWheelhouse, $LinuxWheelhouse | Out-Null

# Every installed dependency is pinned in the lock.  No-deps avoids fetching
# newer transitive packages; the disconnected install and pip check prove the
# wheel set is actually sufficient.
& $Python -m pip download --no-deps -r $WindowsLock -d $WindowsWheelhouse `
  --index-url https://download.pytorch.org/whl/cu128 `
  --extra-index-url https://pypi.org/simple
if ($LASTEXITCODE -ne 0) { throw 'Windows wheel download failed.' }
& $Python -m pip wheel --no-deps --no-index --find-links $WindowsWheelhouse `
  -r $WindowsLock -w $WindowsWheelhouse
if ($LASTEXITCODE -ne 0) { throw 'Windows source distributions could not be built into wheels.' }

$Versions = Get-Content -LiteralPath $VersionsPath -Raw | ConvertFrom-Json
$WslPython = [string]$Versions.wsl_python
function Quote-Bash([string]$Value) { return "'" + $Value.Replace("'", "'`"`'`"`'") + "'" }
function Convert-ToWslPath([string]$Path) {
  $Full = [IO.Path]::GetFullPath($Path)
  if ($Full -notmatch '^([A-Za-z]):\\(.*)$') { throw "Expected local drive path: $Full" }
  return '/mnt/' + $Matches[1].ToLowerInvariant() + '/' + $Matches[2].Replace('\', '/')
}
$LockWsl = Quote-Bash (Convert-ToWslPath $LinuxLock)
$WheelsWsl = Quote-Bash (Convert-ToWslPath $LinuxWheelhouse)
$PythonWsl = Quote-Bash $WslPython
$Command = "$PythonWsl -m pip download --no-deps -r $LockWsl -d $WheelsWsl --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple"
& wsl.exe -d $Distro -- bash -lc $Command
if ($LASTEXITCODE -ne 0) { throw 'FireRed WSL wheel download failed.' }
$BuildCommand = "$PythonWsl -m pip wheel --no-deps --no-index --find-links $WheelsWsl -r $LockWsl -w $WheelsWsl"
& wsl.exe -d $Distro -- bash -lc $BuildCommand
if ($LASTEXITCODE -ne 0) { throw 'FireRed WSL source distributions could not be built into wheels.' }

# The built wheels are the recovery material; source archives are redundant.
Add-Type -AssemblyName Microsoft.VisualBasic
foreach ($Wheelhouse in @($WindowsWheelhouse, $LinuxWheelhouse)) {
  $ResolvedWheelhouse = [IO.Path]::GetFullPath($Wheelhouse)
  if (-not $ResolvedWheelhouse.StartsWith([IO.Path]::GetFullPath($Bundle) + [IO.Path]::DirectorySeparatorChar,
      [StringComparison]::OrdinalIgnoreCase)) { throw 'Wheelhouse path escaped staging bundle.' }
  foreach ($Archive in @(Get-ChildItem -LiteralPath $Wheelhouse -File | Where-Object Extension -ne '.whl')) {
    [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile(
      $Archive.FullName, [Microsoft.VisualBasic.FileIO.UIOption]::OnlyErrorDialogs,
      [Microsoft.VisualBasic.FileIO.RecycleOption]::SendToRecycleBin)
  }
}
Write-Host "Windows wheels: $(@(Get-ChildItem -LiteralPath $WindowsWheelhouse -File).Count); Linux wheels: $(@(Get-ChildItem -LiteralPath $LinuxWheelhouse -File).Count)."
