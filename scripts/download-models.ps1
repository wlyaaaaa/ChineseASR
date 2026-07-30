param(
  [string]$Engine = '',

  [string]$Device = 'cuda:0',

  [switch]$ReceiptOnly,

  [string]$FireRedRevision = '2c5e0f415b9afb8f67cb8b00ea4c54959f70e824'
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Invoke-NoProxy.ps1')
if ($Engine -ne 'fireredasr2-llm') {
  Clear-ProxyEnv
}

function Get-Sha256Hex {
  param(
    [Parameter(Mandatory = $true)]
    [string]$LiteralPath
  )

  $Stream = [System.IO.FileStream]::new(
    $LiteralPath,
    [System.IO.FileMode]::Open,
    [System.IO.FileAccess]::Read,
    [System.IO.FileShare]::Read,
    8MB,
    [System.IO.FileOptions]::SequentialScan
  )
  try {
    $Hasher = [System.Security.Cryptography.SHA256]::Create()
    try {
      $Digest = $Hasher.ComputeHash([System.IO.Stream]$Stream)
      return [Convert]::ToHexString($Digest).ToLowerInvariant()
    }
    finally {
      $Hasher.Dispose()
    }
  }
  finally {
    $Stream.Dispose()
  }
}

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) {
  throw 'Virtual environment not found. Run scripts\install-torch-cu128-direct.ps1 and scripts\setup-core.ps1 first.'
}

$CacheDir = Join-Path $Root 'models\modelscope'
$env:MODELSCOPE_CACHE = $CacheDir
$QwenRevision = 'a04930dbe5419bfee073f7cade734f572689a3a8'

if ($ReceiptOnly -and $Engine -ne 'qwen3-asr-1.7b') {
  throw '-ReceiptOnly currently requires -Engine qwen3-asr-1.7b.'
}

if ($Engine -eq 'fireredasr2-llm') {
  $FireRedDir = Join-Path $Root 'models\firered\FireRedASR2-LLM'
  $env:ZH_ASR_FIRERED_DIR = $FireRedDir
  $env:ZH_ASR_FIRERED_REVISION = $FireRedRevision
  & $Python -c "import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id='FireRedTeam/FireRedASR2-LLM', revision=os.environ['ZH_ASR_FIRERED_REVISION'], local_dir=os.environ['ZH_ASR_FIRERED_DIR'], max_workers=8)"
  if ($LASTEXITCODE -ne 0) {
    throw "Hugging Face prefetch failed for FireRedTeam/FireRedASR2-LLM@$FireRedRevision."
  }

  $RequiredFiles = @(
    'asr_encoder.pth.tar',
    'cmvn.ark',
    'model.pth.tar',
    'Qwen2-7B-Instruct/config.json',
    'Qwen2-7B-Instruct/generation_config.json',
    'Qwen2-7B-Instruct/merges.txt',
    'Qwen2-7B-Instruct/model.safetensors.index.json',
    'Qwen2-7B-Instruct/model-00001-of-00004.safetensors',
    'Qwen2-7B-Instruct/model-00002-of-00004.safetensors',
    'Qwen2-7B-Instruct/model-00003-of-00004.safetensors',
    'Qwen2-7B-Instruct/model-00004-of-00004.safetensors',
    'Qwen2-7B-Instruct/tokenizer_config.json',
    'Qwen2-7B-Instruct/tokenizer.json',
    'Qwen2-7B-Instruct/vocab.json'
  )
  if (@($RequiredFiles | Select-Object -Unique).Count -ne $RequiredFiles.Count) {
    throw 'Internal error: FireRed canonical required-file list contains duplicate paths.'
  }
  $Missing = @($RequiredFiles | Where-Object {
      -not (Test-Path -LiteralPath (Join-Path $FireRedDir $_) -PathType Leaf)
    })
  if ($Missing.Count -gt 0) {
    throw "FireRed download is incomplete. Missing: $($Missing -join ', ')"
  }

  $HashRecords = foreach ($RelativePath in $RequiredFiles) {
    $FullPath = Join-Path $FireRedDir $RelativePath
    $Item = Get-Item -LiteralPath $FullPath
    [ordered]@{
      path = $RelativePath
      bytes = $Item.Length
      sha256 = Get-Sha256Hex -LiteralPath $FullPath
    }
  }
  $Receipt = [ordered]@{
    schema = 'zh_asr.model_receipt.v1'
    repository = 'FireRedTeam/FireRedASR2-LLM'
    revision = $FireRedRevision
    created_utc = [DateTime]::UtcNow.ToString('o')
    files = @($HashRecords)
  }
  $ReceiptPath = Join-Path $FireRedDir 'MODEL_RECEIPT.json'
  $ReceiptTempPath = "$ReceiptPath.partial-$PID"
  try {
    [System.IO.File]::WriteAllText(
      $ReceiptTempPath,
      (($Receipt | ConvertTo-Json -Depth 5 -Compress) + [Environment]::NewLine),
      [System.Text.UTF8Encoding]::new($false)
    )
    Move-Item -LiteralPath $ReceiptTempPath -Destination $ReceiptPath -Force
  }
  finally {
    if (Test-Path -LiteralPath $ReceiptTempPath) {
      Remove-Item -LiteralPath $ReceiptTempPath -Force
    }
  }
  Write-Host "FireRedASR2-LLM weights ready at pinned revision $FireRedRevision`: $FireRedDir"
  Write-Host "SHA-256 receipt: $ReceiptPath"
  Write-Host 'The isolated WSL runtime is loaded only during transcription; Windows warmup is intentionally skipped.'
  exit 0
}

if ($Engine -eq 'qwen3-asr-1.7b') {
  $QwenDir = Join-Path $CacheDir 'Qwen\Qwen3-ASR-1.7B'
  $env:ZH_ASR_QWEN_DIR = $QwenDir
  $env:ZH_ASR_QWEN_REVISION = $QwenRevision
  if (-not (Test-Path -LiteralPath $QwenDir -PathType Container)) {
    if ($ReceiptOnly) {
      throw "Existing Qwen cache not found for receipt-only migration: $QwenDir"
    }
    & $Python -c "import os; from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-ASR-1.7B', revision=os.environ['ZH_ASR_QWEN_REVISION'], local_dir=os.environ['ZH_ASR_QWEN_DIR'], max_workers=8)"
    if ($LASTEXITCODE -ne 0) {
      throw "ModelScope prefetch failed for Qwen/Qwen3-ASR-1.7B@$QwenRevision."
    }
  }

  $ReceiptArgs = @(
    '-m', 'zh_asr.qwen_identity',
    'write-receipt',
    '--model-dir', $QwenDir,
    '--repository', 'Qwen/Qwen3-ASR-1.7B',
    '--revision', $QwenRevision
  )
  & $Python @ReceiptArgs
  if ($LASTEXITCODE -ne 0) {
    throw "Qwen pinned model receipt verification failed for $QwenDir."
  }
  Write-Host "Qwen3-ASR-1.7B weights verified at pinned revision $QwenRevision`: $QwenDir"
  Write-Host "SHA-256 receipt: $(Join-Path $QwenDir 'MODEL_RECEIPT.json')"
  if ($ReceiptOnly) {
    exit 0
  }
}

$Args = @('-m', 'zh_asr', 'warmup', '--device', $Device, '--cache-dir', $CacheDir)
if (-not [string]::IsNullOrWhiteSpace($Engine)) {
  $Args += @('--engine', $Engine)
}
& $Python @Args
if ($LASTEXITCODE -ne 0) {
  throw "Warmup failed for engine '$Engine'."
}
