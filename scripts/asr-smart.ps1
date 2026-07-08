param(
  [Parameter(Mandatory = $true)]
  [string]$Audio,
  [ValidateSet('strict', 'quick', 'long-strict')]
  [string]$Mode = 'strict',
  [string]$Engine = '',
  [string]$PrimaryEngine = '',
  [string]$SecondaryEngine = '',
  [string]$Device = 'cuda:0',
  [string]$OutRoot = '',
  [string]$CacheDir = '',
  [string]$HostName = '127.0.0.1',
  [int]$Port = 8766,
  [int]$WaitSec = 15,
  [int]$StartupTimeoutSec = 30,
  [int]$PollIntervalSec = 2,
  [int]$ChunkSec = 300,
  [int]$OverlapSec = 1,
  [switch]$AllowGpuConflicts,
  [switch]$Force,
  [switch]$Json
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Invoke-NoProxy.ps1')
Clear-ProxyEnv

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) {
  $Python = 'python'
}

$AudioPath = (Resolve-Path $Audio).Path
$StateDir = Join-Path $Root 'outputs\api'
$BaseUrl = "http://$HostName`:$Port"

function Invoke-AsrApi {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Method,
    [Parameter(Mandatory = $true)]
    [string]$Uri,
    [object]$Body = $null
  )

  if ($null -eq $Body) {
    return Invoke-RestMethod -Method $Method -Uri $Uri -TimeoutSec 5
  }

  $JsonBody = $Body | ConvertTo-Json -Depth 8
  return Invoke-RestMethod -Method $Method -Uri $Uri -ContentType 'application/json; charset=utf-8' -Body $JsonBody -TimeoutSec 5
}

function Test-AsrApi {
  try {
    Invoke-AsrApi -Method 'GET' -Uri "$BaseUrl/health" | Out-Null
    return $true
  } catch {
    return $false
  }
}

function Start-AsrApi {
  $Args = @(
    '-m', 'zh_asr',
    'serve',
    '--host', $HostName,
    '--port', "$Port",
    '--state-dir', $StateDir
  )
  Start-Process -FilePath $Python -ArgumentList $Args -WorkingDirectory $Root -WindowStyle Hidden | Out-Null

  $Deadline = (Get-Date).AddSeconds($StartupTimeoutSec)
  while ((Get-Date) -lt $Deadline) {
    if (Test-AsrApi) {
      return
    }
    Start-Sleep -Milliseconds 500
  }
  throw "ASR API did not become ready within $StartupTimeoutSec seconds: $BaseUrl"
}

if (-not (Test-AsrApi)) {
  Start-AsrApi
}

$Payload = [ordered]@{
  audio = $AudioPath
  mode = $Mode
  device = $Device
  chunk_sec = $ChunkSec
  overlap_sec = $OverlapSec
  force = [bool]$Force
  allow_gpu_conflicts = [bool]$AllowGpuConflicts
}
if ($Engine) {
  $Payload.engine = $Engine
}
if ($PrimaryEngine) {
  $Payload.primary_engine = $PrimaryEngine
}
if ($SecondaryEngine) {
  $Payload.secondary_engine = $SecondaryEngine
}
if ($OutRoot) {
  if ([System.IO.Path]::IsPathRooted($OutRoot)) {
    $OutRootPath = $OutRoot
  } else {
    $OutRootPath = Join-Path (Get-Location) $OutRoot
  }
  $OutRootPath = [System.IO.Path]::GetFullPath($OutRootPath)
  New-Item -ItemType Directory -Force -Path $OutRootPath | Out-Null
  $Payload.out_root = $OutRootPath
}
if ($CacheDir) {
  $Payload.cache_dir = (Resolve-Path $CacheDir).Path
}

$Submit = Invoke-AsrApi -Method 'POST' -Uri "$BaseUrl/jobs/transcribe" -Body $Payload
$StatusUri = "$BaseUrl/jobs/$($Submit.job.job_id)"
$Status = $Submit
$Deadline = (Get-Date).AddSeconds($WaitSec)
while ((Get-Date) -lt $Deadline) {
  $Job = $Status.job
  if ($Job.status -in @('succeeded', 'failed', 'blocked', 'canceled')) {
    break
  }
  Start-Sleep -Seconds $PollIntervalSec
  $Status = Invoke-AsrApi -Method 'GET' -Uri $StatusUri
}

$FinalJob = $Status.job
$Result = [ordered]@{
  status = $FinalJob.status
  job_id = $FinalJob.job_id
  out_dir = $FinalJob.out_dir
  outputs = $FinalJob.outputs
  message = $FinalJob.message
  deduplicated = [bool]$Submit.deduplicated
  next_status_command = "Invoke-RestMethod -Uri '$StatusUri'"
}

if ($Json) {
  $Result | ConvertTo-Json -Depth 8
} else {
  Write-Host "Status: $($Result.status)"
  Write-Host "Job: $($Result.job_id)"
  Write-Host "Output: $($Result.out_dir)"
  if ($Result.outputs) {
    $Result.outputs.PSObject.Properties | ForEach-Object {
      Write-Host "$($_.Name): $($_.Value)"
    }
  }
  if ($Result.message) {
    Write-Host "Message: $($Result.message)"
  }
  Write-Host "Next: $($Result.next_status_command)"
}

if ($FinalJob.status -in @('failed', 'blocked', 'canceled')) {
  exit 1
}
