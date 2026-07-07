param(
  [string]$Audio = '',
  [string]$HostName = '127.0.0.1',
  [int]$Port = 8765,
  [int]$WaitSec = 300,
  [int]$StartupTimeoutSec = 120,
  [int]$PollIntervalSec = 5,
  [switch]$Json
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Invoke-NoProxy.ps1')
Clear-ProxyEnv

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if ($Audio) {
  $AudioPath = (Resolve-Path $Audio).Path
} else {
  $AudioPath = Join-Path $Root 'models\modelscope\iic\SenseVoiceSmall\example\zh.mp3'
  if (-not (Test-Path $AudioPath)) {
    throw "Default smoke audio not found: $AudioPath"
  }
}

$AsrSmart = Join-Path $PSScriptRoot 'asr-smart.ps1'
$ResultJson = & $AsrSmart `
  -Audio $AudioPath `
  -Mode strict `
  -HostName $HostName `
  -Port $Port `
  -WaitSec $WaitSec `
  -StartupTimeoutSec $StartupTimeoutSec `
  -PollIntervalSec $PollIntervalSec `
  -Force `
  -Json

$Result = $ResultJson | ConvertFrom-Json
if ($Result.status -ne 'succeeded') {
  throw "asr-smart strict smoke did not succeed: status=$($Result.status), job=$($Result.job_id), message=$($Result.message)"
}

$RequiredOutputs = @('final', 'audit', 'audit_json', 'primary_raw_json', 'secondary_raw_json')
foreach ($Name in $RequiredOutputs) {
  $Value = $Result.outputs.$Name
  if (-not $Value) {
    throw "Missing smoke output field: $Name"
  }
  if (-not (Test-Path $Value)) {
    throw "Missing smoke output file for ${Name}: $Value"
  }
}

if ($Json) {
  $Result | ConvertTo-Json -Depth 8
} else {
  Write-Host "Status: $($Result.status)"
  Write-Host "Job: $($Result.job_id)"
  Write-Host "Output: $($Result.out_dir)"
  Write-Host "Final: $($Result.outputs.final)"
  Write-Host "Audit: $($Result.outputs.audit)"
  Write-Host "Primary raw JSON: $($Result.outputs.primary_raw_json)"
  Write-Host "Secondary raw JSON: $($Result.outputs.secondary_raw_json)"
}
