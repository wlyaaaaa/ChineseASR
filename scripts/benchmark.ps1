param(
  [Parameter(Mandatory = $true)]
  [string]$AudioDir,

  [Parameter(Mandatory = $true)]
  [string]$TruthDir,

  [string]$OutDir = 'E:\ChineseASR\outputs\benchmark',
  [string]$Device = 'cuda:0',
  [string]$PrimaryEngine = '',
  [string]$SecondaryEngine = '',
  [switch]$Force,
  [switch]$FailOnFindings
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
  'benchmark',
  '--audio-dir', $AudioDir,
  '--truth-dir', $TruthDir,
  '--out-dir', $OutDir,
  '--device', $Device,
  '--cache-dir', (Join-Path $Root 'models\modelscope')
)
if ($Force) {
  $CliArgs += '--force'
}
if ($FailOnFindings) {
  $CliArgs += '--fail-on-findings'
}
if (-not [string]::IsNullOrWhiteSpace($PrimaryEngine)) {
  $CliArgs += @('--primary-engine', $PrimaryEngine)
}
if (-not [string]::IsNullOrWhiteSpace($SecondaryEngine)) {
  $CliArgs += @('--secondary-engine', $SecondaryEngine)
}

$env:ZH_ASR_WRAPPER = 'scripts\benchmark.ps1'
& $Python @CliArgs
if ($LASTEXITCODE -ne 0) {
  throw "Benchmark failed with exit code $LASTEXITCODE."
}
