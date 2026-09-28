param(
  [string]$RuntimeRoot = 'E:\Projects\Tools\ChineseASR',
  [string]$BackupRoot = 'E:\Projects\Tools\ChineseASR\offline',
  [string]$StagingDir = '',
  [string]$SmokeRoot = 'E:\Cache\Codex\Temp\ChineseASR-offline-smoke',
  [string]$Distro = 'Ubuntu',
  [switch]$SkipLinuxSmoke
)

$ErrorActionPreference = 'Stop'
$BackupRoot = [IO.Path]::GetFullPath($BackupRoot)
$RuntimeRoot = [IO.Path]::GetFullPath($RuntimeRoot)
if (-not $BackupRoot.StartsWith($RuntimeRoot + [IO.Path]::DirectorySeparatorChar,
    [StringComparison]::OrdinalIgnoreCase) -or
    [IO.Path]::GetPathRoot($BackupRoot) -notmatch '^[Ee]:\\$') {
  throw 'Backup root must be an E: directory within the installed ChineseASR project.'
}
New-Item -ItemType Directory -Force -Path $BackupRoot | Out-Null
$Current = Join-Path $BackupRoot 'current'
$RunId = [guid]::NewGuid().ToString('N')
$Stage = if ($StagingDir) { [IO.Path]::GetFullPath($StagingDir) }
         else { Join-Path $BackupRoot "staging-$RunId" }
if (-not $Stage.StartsWith($BackupRoot + [IO.Path]::DirectorySeparatorChar,
    [StringComparison]::OrdinalIgnoreCase) -or $Stage -eq $Current) {
  throw 'Staging directory must be separate and inside the backup root.'
}
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $RuntimeRoot '.venv\Scripts\python.exe'
if (-not $StagingDir) {
  & (Join-Path $PSScriptRoot 'export-lock.ps1') -RuntimeRoot $RuntimeRoot -Bundle $Stage -Distro $Distro
  & (Join-Path $PSScriptRoot 'build-wheelhouse.ps1') -RuntimeRoot $RuntimeRoot -Bundle $Stage -Distro $Distro
  & $Python -B (Join-Path $PSScriptRoot 'offline_bundle.py') copy-models `
    --runtime-root $RuntimeRoot --config (Join-Path $RuntimeRoot 'configs\models.yaml') --bundle $Stage
  if ($LASTEXITCODE -ne 0) { throw 'Copying configured models failed.' }
  & $Python -B (Join-Path $PSScriptRoot 'offline_bundle.py') compare-models `
    --runtime-root $RuntimeRoot --config (Join-Path $RuntimeRoot 'configs\models.yaml') --bundle $Stage
  if ($LASTEXITCODE -ne 0) { throw 'Source and offline model hashes differ.' }
  & (Join-Path $PSScriptRoot 'verify-wheelhouse.ps1') -RuntimeRoot $RuntimeRoot -Bundle $Stage -Seal
}
& (Join-Path $PSScriptRoot 'verify-wheelhouse.ps1') -RuntimeRoot $RuntimeRoot -Bundle $Stage

$SmokeRoot = [IO.Path]::GetFullPath($SmokeRoot)
if ([IO.Path]::GetPathRoot($SmokeRoot) -notmatch '^[Ee]:\\$') {
  throw 'Smoke directory must be on E:.'
}
New-Item -ItemType Directory -Force -Path $SmokeRoot | Out-Null
$WindowsSmoke = Join-Path $SmokeRoot "windows-$RunId"
$LinuxSmoke = "/tmp/chineseasr-offline-$RunId"
$InstallParameters = @{
  ProjectRoot = $ProjectRoot
  RuntimeRoot = $RuntimeRoot
  Bundle = $Stage
  Venv = $WindowsSmoke
  Distro = $Distro
}
if (-not $SkipLinuxSmoke) {
  $InstallParameters.VerifyLinux = $true
  $InstallParameters.LinuxVenv = $LinuxSmoke
}
& (Join-Path $PSScriptRoot 'install-offline.ps1') @InstallParameters

# Keep the verified new copy intact before retiring the former copy.
$Old = Join-Path $BackupRoot "previous-$RunId"
if (Test-Path -LiteralPath $Current) {
  Move-Item -LiteralPath $Current -Destination $Old
}
try {
  Move-Item -LiteralPath $Stage -Destination $Current
}
catch {
  if (Test-Path -LiteralPath $Old) { Move-Item -LiteralPath $Old -Destination $Current }
  throw
}

function Move-ToRecycleBin([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path)) { return }
  Add-Type -AssemblyName Microsoft.VisualBasic
  try {
    [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteDirectory(
      $Path, [Microsoft.VisualBasic.FileIO.UIOption]::OnlyErrorDialogs,
      [Microsoft.VisualBasic.FileIO.RecycleOption]::SendToRecycleBin)
  }
  catch { Write-Warning "Could not move temporary/previous data to Recycle Bin; retained at $Path : $_" }
}
Move-ToRecycleBin $Old
Move-ToRecycleBin $WindowsSmoke
if (-not $SkipLinuxSmoke) {
  $LinuxTrashCommand = "if [ -e '$LinuxSmoke' ]; then gio trash -- '$LinuxSmoke'; fi; if [ -e '$LinuxSmoke-source' ]; then gio trash -- '$LinuxSmoke-source'; fi"
  & wsl.exe -d $Distro -- bash -lc $LinuxTrashCommand
  if ($LASTEXITCODE -ne 0) {
    Write-Warning "Could not move WSL smoke directories to Linux Trash; retained at $LinuxSmoke and $LinuxSmoke-source"
  }
}
Write-Host "Verified offline backup published: $Current"
$global:LASTEXITCODE = 0
