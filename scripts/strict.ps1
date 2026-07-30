param(
  [Parameter(Mandatory = $true)]
  [string]$Audio,

  [string]$Device = 'cuda:0',
  [string]$OutDir = '',
  [string]$PrimaryEngine = '',
  [string]$SecondaryEngine = ''
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Invoke-NoProxy.ps1')
Clear-ProxyEnv

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) {
  throw 'Virtual environment not found. Run scripts\install-torch-cu128-direct.ps1 and scripts\setup-core.ps1 first.'
}
if ([string]::IsNullOrWhiteSpace($OutDir)) {
  $OutDir = Join-Path $Root 'outputs\strict'
}

$Args = @(
  '-Audio', $Audio,
  '-Mode', 'strict',
  '-Device', $Device,
  '-OutRoot', $OutDir,
  '-CacheDir', (Join-Path $Root 'models\modelscope'),
  '-WaitSec', '21600',
  '-StartupTimeoutSec', '120'
)
if (-not [string]::IsNullOrWhiteSpace($PrimaryEngine)) {
  $Args += @('-PrimaryEngine', $PrimaryEngine)
}
if (-not [string]::IsNullOrWhiteSpace($SecondaryEngine)) {
  $Args += @('-SecondaryEngine', $SecondaryEngine)
}

& (Join-Path $PSScriptRoot 'asr-smart.ps1') @Args
