param(
  [string]$Engine = '',

  [string]$Device = 'cuda:0'
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Invoke-NoProxy.ps1')
Clear-ProxyEnv

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) {
  throw 'Virtual environment not found. Run scripts\install-torch-cu128-direct.ps1 and scripts\setup-core.ps1 first.'
}

$CacheDir = Join-Path $Root 'models\modelscope'
$env:MODELSCOPE_CACHE = $CacheDir
$Args = @('-m', 'zh_asr', 'warmup', '--device', $Device, '--cache-dir', $CacheDir)
if (-not [string]::IsNullOrWhiteSpace($Engine)) {
  $Args += @('--engine', $Engine)
}
& $Python @Args
