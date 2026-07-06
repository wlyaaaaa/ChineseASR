$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Invoke-NoProxy.ps1')
Clear-ProxyEnv

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Venv = Join-Path $Root '.venv'
if (-not (Test-Path $Venv)) {
  python -m venv $Venv
}

$Python = Join-Path $Venv 'Scripts\python.exe'
& $Python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
& $Python -m pip install -r (Join-Path $Root 'requirements-core.txt') -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
& $Python -m pip install -e $Root --no-deps
& $Python -m zh_asr doctor

