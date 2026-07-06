param(
  [switch]$Nightly,
  [string]$Wheelhouse = ''
)

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

if ($Wheelhouse) {
  & $Python -m pip install --no-index --find-links $Wheelhouse torch torchaudio
} elseif ($Nightly) {
  & $Python -m pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --trusted-host download.pytorch.org
} else {
  & $Python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128 --trusted-host download.pytorch.org
}

& $Python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

