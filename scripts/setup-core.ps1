$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Invoke-NoProxy.ps1')
Clear-ProxyEnv

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Venv = Join-Path $Root '.venv'
if (-not (Test-Path $Venv)) {
  python -m venv $Venv
}

$Python = Join-Path $Venv 'Scripts\python.exe'
$MirrorIndex = 'https://pypi.tuna.tsinghua.edu.cn/simple'
$OfficialIndex = 'https://pypi.org/simple'

function Invoke-PipInstallWithFallback {
  param([Parameter(Mandatory = $true)][string[]]$PipArguments)

  & $Python -m pip @PipArguments -i $MirrorIndex --trusted-host pypi.tuna.tsinghua.edu.cn
  if ($LASTEXITCODE -eq 0) {
    return
  }

  Write-Warning "Tsinghua PyPI mirror could not satisfy the request; retrying official PyPI."
  & $Python -m pip @PipArguments -i $OfficialIndex
  if ($LASTEXITCODE -ne 0) {
    throw "pip failed against both configured indexes: $($PipArguments -join ' ')"
  }
}

Invoke-PipInstallWithFallback -PipArguments @('install', '--upgrade', 'pip')
Invoke-PipInstallWithFallback -PipArguments @('install', '-r', (Join-Path $Root 'requirements-core.txt'))
& $Python -m pip install -e $Root --no-deps
if ($LASTEXITCODE -ne 0) {
  throw 'Editable ChineseASR install failed.'
}
& $Python -m pip check
if ($LASTEXITCODE -ne 0) {
  throw 'ChineseASR environment has broken requirements.'
}
& $Python -m zh_asr doctor
if ($LASTEXITCODE -ne 0) {
  throw 'ChineseASR doctor failed.'
}
