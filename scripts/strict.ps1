param(
  [Parameter(Mandatory = $true)]
  [string]$Audio,

  [string]$Device = 'cuda:0',
  [string]$OutDir = 'E:\ChineseASR\outputs\strict'
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Invoke-NoProxy.ps1')
Clear-ProxyEnv

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) {
  throw 'Virtual environment not found. Run scripts\install-torch-cu128-direct.ps1 and scripts\setup-core.ps1 first.'
}

& $Python -m zh_asr strict $Audio --primary-engine sensevoice --secondary-engine paraformer --device $Device --out-dir $OutDir --cache-dir (Join-Path $Root 'models\modelscope')

