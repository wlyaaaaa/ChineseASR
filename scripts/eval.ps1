param(
  [string]$CorpusDir = 'E:\ChineseASR\eval\corpus\builtin',
  [string]$OutDir = 'E:\ChineseASR\outputs\eval',
  [string]$Device = 'cuda:0',
  [string]$PrimaryEngine = '',
  [string]$SecondaryEngine = '',
  [switch]$Generate,
  [switch]$GenerateOnly,
  [switch]$NoTts,
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
  'eval',
  '--corpus-dir', $CorpusDir,
  '--out-dir', $OutDir,
  '--device', $Device,
  '--cache-dir', (Join-Path $Root 'models\modelscope')
)
if ($Generate) {
  $CliArgs += '--generate'
}
if ($GenerateOnly) {
  $CliArgs += '--generate-only'
}
if ($NoTts) {
  $CliArgs += '--no-tts'
}
if ($Force) {
  $CliArgs += '--force'
}
if (-not [string]::IsNullOrWhiteSpace($PrimaryEngine)) {
  $CliArgs += @('--primary-engine', $PrimaryEngine)
}
if (-not [string]::IsNullOrWhiteSpace($SecondaryEngine)) {
  $CliArgs += @('--secondary-engine', $SecondaryEngine)
}

& $Python @CliArgs
if ($LASTEXITCODE -ne 0) {
  throw "Evaluation failed with exit code $LASTEXITCODE."
}
