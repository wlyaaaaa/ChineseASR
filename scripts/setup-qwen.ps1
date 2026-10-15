$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Invoke-NoProxy.ps1')

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) {
  throw 'Virtual environment not found. Run scripts\install-torch-cu128-direct.ps1 and scripts\setup-core.ps1 first.'
}

Invoke-NoProxyCommand -FilePath $Python -ArgumentList @('-m', 'pip', 'install', '-r', (Join-Path $Root 'requirements-qwen.txt'), '-i', 'https://pypi.tuna.tsinghua.edu.cn/simple', '--trusted-host', 'pypi.tuna.tsinghua.edu.cn')
if ($LASTEXITCODE -ne 0) {
  throw 'Qwen dependency installation failed.'
}
Invoke-NoProxyCommand -FilePath $Python -ArgumentList @('-m', 'pip', 'install', '-e', $Root, '--no-deps')
if ($LASTEXITCODE -ne 0) {
  throw 'Editable ChineseASR install failed.'
}
Invoke-NoProxyCommand -FilePath $Python -ArgumentList @('-m', 'zh_asr', 'doctor')
exit $LASTEXITCODE
