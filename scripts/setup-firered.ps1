param(
  [string]$Distro = 'Ubuntu',
  [string]$InstallRoot = '/opt/chineseasr/firered',
  [string]$SourceDirectory = '',
  [string]$SourceRef = '4e7d9aaf4482a47cec1724807026b9b151926eb5',
  [string]$TorchVersion = '2.10.0+cu128',
  [string]$TorchAudioVersion = '2.10.0+cu128',
  [string]$TorchIndexUrl = 'https://download.pytorch.org/whl/cu128',
  [string]$PypiIndexUrl = 'https://pypi.org/simple',
  [switch]$SkipTorch,
  [switch]$SkipSourceClone
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Invoke-NoProxy.ps1')
Clear-ProxyEnv

function Quote-BashLiteral {
  param([Parameter(Mandatory = $true)][string]$Value)
  return "'" + $Value.Replace("'", "'`"`'`"`'") + "'"
}

function Convert-ToWslPath {
  param([Parameter(Mandatory = $true)][string]$WindowsPath)
  $FullPath = [System.IO.Path]::GetFullPath($WindowsPath)
  if ($FullPath -notmatch '^([A-Za-z]):[\\/](.*)$') {
    throw "Only absolute local Windows drive paths can be translated for WSL: $FullPath"
  }
  $Drive = $Matches[1].ToLowerInvariant()
  $Tail = $Matches[2].Replace('\', '/')
  return "/mnt/$Drive/$Tail"
}

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Requirements = Join-Path $Root 'requirements-firered.txt'
$ResolvedSourceDirectory = if ($SourceDirectory) {
  [System.IO.Path]::GetFullPath($SourceDirectory, $Root)
} else {
  Join-Path $Root 'models\firered\FireRedASR2S'
}
if (-not (Test-Path -LiteralPath $Requirements)) {
  throw "Requirements file not found: $Requirements"
}
if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
  throw 'wsl.exe is not available. Install WSL2 and an Ubuntu distribution first.'
}

$ProbeOutput = & wsl.exe -d $Distro -- sh -lc 'printf firered-wsl-ok'
if ($LASTEXITCODE -ne 0 -or (($ProbeOutput -join '').Trim() -ne 'firered-wsl-ok')) {
  throw "WSL distribution '$Distro' is unavailable or failed to start."
}

$RequirementsWsl = Convert-ToWslPath $Requirements
$SourceWsl = Convert-ToWslPath $ResolvedSourceDirectory

$InstallRootQ = Quote-BashLiteral $InstallRoot
$RequirementsQ = Quote-BashLiteral $RequirementsWsl
$SourceDirQ = Quote-BashLiteral $SourceWsl
$SourceRefQ = Quote-BashLiteral $SourceRef
$TorchVersionQ = Quote-BashLiteral $TorchVersion
$TorchAudioVersionQ = Quote-BashLiteral $TorchAudioVersion
$TorchIndexQ = Quote-BashLiteral $TorchIndexUrl
$PypiIndexQ = Quote-BashLiteral $PypiIndexUrl
$SkipTorchValue = if ($SkipTorch) { '1' } else { '0' }
$SkipCloneValue = if ($SkipSourceClone) { '1' } else { '0' }

$Bash = @"
set -euo pipefail
INSTALL_ROOT=$InstallRootQ
INSTALL_ROOT="`${INSTALL_ROOT/#\~/`$HOME}"
REQ=$RequirementsQ
SOURCE_DIR=$SourceDirQ
SOURCE_REF=$SourceRefQ
TORCH_VERSION=$TorchVersionQ
TORCHAUDIO_VERSION=$TorchAudioVersionQ
TORCH_INDEX=$TorchIndexQ
PYPI_INDEX=$PypiIndexQ
SKIP_TORCH=$SkipTorchValue
SKIP_CLONE=$SkipCloneValue
VENV_DIR="`$INSTALL_ROOT/.venv"
SOURCE_REPO=https://github.com/FireRedTeam/FireRedASR2S.git

for command_name in python3 git; do
  if ! command -v "`$command_name" >/dev/null 2>&1; then
    echo "Missing WSL prerequisite: `$command_name" >&2
    exit 20
  fi
done

mkdir -p "`$INSTALL_ROOT"
if [ ! -d "`$SOURCE_DIR/.git" ]; then
  if [ "`$SKIP_CLONE" = "1" ]; then
    echo "Pinned FireRed source checkout is absent and -SkipSourceClone was requested: `$SOURCE_DIR" >&2
    exit 21
  fi
  mkdir -p "`$(dirname "`$SOURCE_DIR")"
  git clone --filter=blob:none --no-checkout "`$SOURCE_REPO" "`$SOURCE_DIR"
fi
ACTUAL_SOURCE_REMOTE="`$(git -C "`$SOURCE_DIR" remote get-url origin)"
if [ "`$ACTUAL_SOURCE_REMOTE" != "`$SOURCE_REPO" ]; then
  echo "FireRed source remote mismatch: expected `$SOURCE_REPO, got `$ACTUAL_SOURCE_REMOTE" >&2
  exit 24
fi
if [ -n "`$(git -c core.autocrlf=true -C "`$SOURCE_DIR" status --porcelain)" ]; then
  echo "Refusing to change a dirty FireRedASR2S checkout: `$SOURCE_DIR" >&2
  exit 23
fi
if [ "`$SKIP_CLONE" != "1" ]; then
  git -C "`$SOURCE_DIR" fetch --depth=1 origin "`$SOURCE_REF"
  git -C "`$SOURCE_DIR" checkout --detach FETCH_HEAD
fi
ACTUAL_SOURCE_COMMIT="`$(git -C "`$SOURCE_DIR" rev-parse HEAD)"
EXPECTED_SOURCE_COMMIT="`$(git -C "`$SOURCE_DIR" rev-parse "`$SOURCE_REF^{commit}" 2>/dev/null || true)"
if [ -z "`$EXPECTED_SOURCE_COMMIT" ] || [ "`$ACTUAL_SOURCE_COMMIT" != "`$EXPECTED_SOURCE_COMMIT" ]; then
  echo "FireRed source revision mismatch: expected `$SOURCE_REF, got `$ACTUAL_SOURCE_COMMIT" >&2
  exit 21
fi

if [ ! -x "`$VENV_DIR/bin/python" ]; then
  if ! python3 -m venv "`$VENV_DIR"; then
    echo "Could not create the WSL virtual environment. Install python3-venv in '$Distro'." >&2
    exit 22
  fi
fi

"`$VENV_DIR/bin/python" -m pip install --upgrade pip -i "`$PYPI_INDEX"
if [ "`$SKIP_TORCH" != "1" ]; then
  "`$VENV_DIR/bin/python" -m pip install \
    "torch==`$TORCH_VERSION" "torchaudio==`$TORCHAUDIO_VERSION" \
    --index-url "`$TORCH_INDEX" --trusted-host download.pytorch.org
fi
"`$VENV_DIR/bin/python" -m pip install -r "`$REQ" -i "`$PYPI_INDEX"

PYTHONPATH="`$SOURCE_DIR" "`$VENV_DIR/bin/python" - <<'PY'
import numpy
import torch
import transformers
import fireredasr2s
import kaldi_native_fbank
from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config
if not torch.cuda.is_available():
    raise SystemExit("FireRed CUDA verification failed: torch.cuda.is_available() is false")
print("firered_import=ok")
print("torch_version=" + torch.__version__)
print("transformers_version=" + transformers.__version__)
print("numpy_version=" + numpy.__version__)
print("cuda_available=true")
print("cuda_device=" + torch.cuda.get_device_name(0))
print("cuda_bf16=" + str(torch.cuda.is_bf16_supported()).lower())
PY

printf 'install_root=%s\n' "`$INSTALL_ROOT"
printf 'source_dir=%s\n' "`$SOURCE_DIR"
printf 'python=%s\n' "`$VENV_DIR/bin/python"
printf 'source_commit=%s\n' "`$ACTUAL_SOURCE_COMMIT"
echo "model_downloaded=false"
"@

$Bash | & wsl.exe -d $Distro -- bash -s --
if ($LASTEXITCODE -ne 0) {
  throw "FireRed isolated runtime setup failed with exit code $LASTEXITCODE."
}

$NormalizedRoot = $InstallRoot
if ($NormalizedRoot.StartsWith('~/')) {
  $WslHome = (& wsl.exe -d $Distro -- sh -lc 'printf %s "$HOME"' | Out-String).Trim()
  $NormalizedRoot = $WslHome.TrimEnd('/') + '/' + $NormalizedRoot.Substring(2)
}
$PythonWsl = $NormalizedRoot.TrimEnd('/') + '/.venv/bin/python'

Write-Host ''
Write-Host 'FireRed runtime prepared. No model weights were downloaded and configs/models.yaml was not changed.'
Write-Host 'The output below matches the optional engine entry shipped in configs/models.yaml:'
Write-Host ''
Write-Host '  adapter: firered-worker'
Write-Host '  model: FireRedTeam/FireRedASR2-LLM'
Write-Host '  options:'
Write-Host '    runtime: wsl'
Write-Host "    wsl_distribution: $Distro"
Write-Host "    python_path: '$PythonWsl'"
Write-Host '    source_dir: models/firered/FireRedASR2S'
Write-Host '    model_dir: models/firered/FireRedASR2-LLM'
Write-Host '    max_audio_sec: 40'
Write-Host '    recommended_chunk_sec: 35'
Write-Host '    timeout_sec: 900'
