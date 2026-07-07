param(
  [Parameter(Mandatory = $true)]
  [string]$InputDir,

  [ValidateSet('strict', 'quick')]
  [string]$Mode = 'strict',

  [string]$Device = 'cuda:0',
  [string]$OutDir = 'E:\ChineseASR\outputs\batch',
  [string]$Engine = '',
  [string]$PrimaryEngine = '',
  [string]$SecondaryEngine = '',
  [switch]$Force
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Invoke-NoProxy.ps1')
Clear-ProxyEnv

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) {
  throw 'Virtual environment not found. Run scripts\install-torch-cu128-direct.ps1 and scripts\setup-core.ps1 first.'
}

$CliArgs = @(
  '-m', 'zh_asr',
  'batch', $InputDir,
  '--mode', $Mode,
  '--device', $Device,
  '--out-dir', $OutDir,
  '--cache-dir', (Join-Path $Root 'models\modelscope')
)
if (-not [string]::IsNullOrWhiteSpace($Engine)) {
  $CliArgs += @('--engine', $Engine)
}
if (-not [string]::IsNullOrWhiteSpace($PrimaryEngine)) {
  $CliArgs += @('--primary-engine', $PrimaryEngine)
}
if (-not [string]::IsNullOrWhiteSpace($SecondaryEngine)) {
  $CliArgs += @('--secondary-engine', $SecondaryEngine)
}
if ($Force) {
  $CliArgs += '--force'
}

& $Python @CliArgs
if ($LASTEXITCODE -ne 0) {
  throw "Folder transcription failed with exit code $LASTEXITCODE."
}
