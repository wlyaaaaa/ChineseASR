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
  '-m', 'zh_asr',
  'strict', $Audio,
  '--device', $Device,
  '--out-dir', $OutDir,
  '--cache-dir', (Join-Path $Root 'models\modelscope')
)
if (-not [string]::IsNullOrWhiteSpace($PrimaryEngine)) {
  $Args += @('--primary-engine', $PrimaryEngine)
}
if (-not [string]::IsNullOrWhiteSpace($SecondaryEngine)) {
  $Args += @('--secondary-engine', $SecondaryEngine)
}

& $Python @Args
