$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Invoke-NoProxy.ps1')
Clear-ProxyEnv

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) {
  $Python = 'python'
  $env:PYTHONPATH = Join-Path $Root 'src'
}

Write-Host '== WinHTTP proxy =='
netsh winhttp show proxy
Write-Host ''
Write-Host '== GPU =='
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
Write-Host ''
Write-Host '== ASR project =='
& $Python -m zh_asr doctor

