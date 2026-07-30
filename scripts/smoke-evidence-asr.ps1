param(
  [Parameter(Mandatory = $true)]
  [string]$Audio,

  [string]$OutRoot = '',
  [string]$HostName = '127.0.0.1',
  [int]$Port = 18666,
  [int]$WaitSec = 3600,
  [int]$StartupTimeoutSec = 120,
  [int]$PollIntervalSec = 5,
  [switch]$Json
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Invoke-NoProxy.ps1')
Clear-ProxyEnv

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$AudioPath = (Resolve-Path $Audio).Path
if ([string]::IsNullOrWhiteSpace($OutRoot)) {
  $OutRoot = Join-Path $Root 'outputs\evidence-smoke'
}

$ResultJson = & (Join-Path $PSScriptRoot 'asr-smart.ps1') `
  -Audio $AudioPath `
  -Mode long-strict `
  -PrimaryEngine fireredasr2-llm `
  -SecondaryEngine qwen3-asr-1.7b `
  -OutRoot $OutRoot `
  -HostName $HostName `
  -Port $Port `
  -WaitSec $WaitSec `
  -StartupTimeoutSec $StartupTimeoutSec `
  -PollIntervalSec $PollIntervalSec `
  -ChunkSec 300 `
  -OverlapSec 1 `
  -Force `
  -Json

$Result = $ResultJson | ConvertFrom-Json
if ($Result.status -ne 'succeeded') {
  throw "Evidence ASR smoke did not succeed: status=$($Result.status), job=$($Result.job_id), message=$($Result.message)"
}
if ($Result.evidence_status -ne 'verified') {
  $Failures = $Result.evidence_failures | ConvertTo-Json -Depth 8 -Compress
  throw "Evidence ASR smoke is not verified: evidence_status=$($Result.evidence_status), failures=$Failures"
}
if (-not $Result.outputs.manifest -or -not (Test-Path -LiteralPath $Result.outputs.manifest -PathType Leaf)) {
  throw 'Evidence ASR smoke did not return a readable manifest.'
}

$Manifest = Get-Content -LiteralPath $Result.outputs.manifest -Raw -Encoding UTF8 | ConvertFrom-Json
if ($Manifest.evidence_status -ne 'verified' -or @($Manifest.chunks).Count -eq 0) {
  throw "Evidence manifest is not a non-empty verified run: $($Result.outputs.manifest)"
}

$RequiredOutputKeys = @(
  'final',
  'audit',
  'audit_json',
  'review_json',
  'receipt',
  'primary_json',
  'secondary_json'
)
foreach ($Chunk in @($Manifest.chunks)) {
  if ($Chunk.status -ne 'succeeded' -or $Chunk.evidence_status -ne 'verified') {
    throw "Chunk $($Chunk.chunk_id) is not succeeded + verified."
  }
  foreach ($Key in $RequiredOutputKeys) {
    $PathValue = $Chunk.outputs.$Key
    if (-not $PathValue -or -not (Test-Path -LiteralPath $PathValue -PathType Leaf)) {
      throw "Chunk $($Chunk.chunk_id) is missing required evidence artifact: $Key"
    }
  }

  $Audit = Get-Content -LiteralPath $Chunk.outputs.audit_json -Raw -Encoding UTF8 | ConvertFrom-Json
  if ($Audit.status -eq 'engine_failure') {
    throw "Chunk $($Chunk.chunk_id) contains engine_failure."
  }
  if (
    [string]::IsNullOrWhiteSpace([string]$Audit.bundle_receipt_reference) -or
    [System.IO.Path]::IsPathRooted([string]$Audit.bundle_receipt_reference)
  ) {
    throw "Chunk $($Chunk.chunk_id) does not use a portable bundle receipt reference."
  }
  $EngineEvidence = @($Audit.engine_evidence)
  if (
    $EngineEvidence.Count -ne 2 -or
    @($EngineEvidence | Where-Object {
        $_.execution_status -ne 'succeeded' -or
        [string]::IsNullOrWhiteSpace([string]$_.text) -or
        [string]::IsNullOrWhiteSpace([string]$_.raw_result_reference) -or
        [System.IO.Path]::IsPathRooted([string]$_.raw_result_reference)
      }).Count -gt 0
  ) {
    throw "Chunk $($Chunk.chunk_id) does not contain two successful non-empty portable engine records."
  }

  $FireRedRaw = Get-Content -LiteralPath $Chunk.outputs.primary_json -Raw -Encoding UTF8 | ConvertFrom-Json
  if (
    [string]::IsNullOrWhiteSpace([string]$FireRedRaw.text) -or
    $null -ne $FireRedRaw.error
  ) {
    throw "Chunk $($Chunk.chunk_id) FireRed raw result is empty or contains an error."
  }
  $LoadDtype = [string]$FireRedRaw._zh_asr_runtime.llm_initial_load_dtype
  if ($LoadDtype -notin @('bfloat16', 'float16')) {
    throw "Chunk $($Chunk.chunk_id) FireRed runtime dtype is not evidence-grade half precision: $LoadDtype"
  }
}

if ($Json) {
  $Result | ConvertTo-Json -Depth 10
} else {
  Write-Host 'Evidence ASR smoke: PASS'
  Write-Host "Job: $($Result.job_id)"
  Write-Host "Manifest: $($Result.outputs.manifest)"
  Write-Host "Chunks: $(@($Manifest.chunks).Count)"
}
