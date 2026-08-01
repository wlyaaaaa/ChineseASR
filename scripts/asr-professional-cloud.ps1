#requires -Version 7.2

[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string] $Audio,

    [switch] $Important,

    [switch] $CloudUploadAuthorized,

    [ValidateRange(1, 180)]
    [int] $ChunkSec = 180,

    [ValidateRange(0, 179)]
    [int] $OverlapSec = 1,

    [string] $RequestRoot = 'E:\Projects\Tools\ChineseASR\outputs\cloud-jobs',

    [switch] $Json
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$utf8NoBom = [Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8NoBom
$OutputEncoding = $utf8NoBom

$brokerPath = 'C:\ProgramData\PCConfig\AuthorityHost\tools\Invoke-SecretBroker.ps1'
$brokerTarget = 'qwen-audio3-asr-important-once'
$canonicalRequestRoot = [IO.Path]::GetFullPath(
    'E:\Projects\Tools\ChineseASR\outputs\cloud-jobs'
)

function Write-BoundedReceipt {
    param(
        [Parameter(Mandatory)][string] $Status,
        [Parameter(Mandatory)][string] $ErrorCode,
        [Parameter(Mandatory)][int] $ExitCode
    )

    $receipt = [ordered]@{
        schema = 'chineseasr.qwen-audio3-important-result.v1'
        status = $Status
        error_code = $ErrorCode
        important_only = $true
        cloud_upload_performed = $false
        plaintext_returned = $false
        secret_returned = $false
    }
    if ($Json) {
        $receipt | ConvertTo-Json -Depth 8 -Compress | Write-Output
    }
    else {
        $receipt
    }
    exit $ExitCode
}

if (-not $Important) {
    Write-BoundedReceipt -Status 'blocked' -ErrorCode 'importance_required' -ExitCode 2
}
if (-not $CloudUploadAuthorized) {
    Write-BoundedReceipt `
        -Status 'blocked' `
        -ErrorCode 'cloud_upload_authorization_required' `
        -ExitCode 2
}
if ($OverlapSec -ge $ChunkSec) {
    Write-BoundedReceipt -Status 'blocked' -ErrorCode 'chunk_policy_invalid' -ExitCode 2
}

$resolvedRequestRoot = [IO.Path]::GetFullPath($RequestRoot)
if ($resolvedRequestRoot -cne $canonicalRequestRoot) {
    Write-BoundedReceipt -Status 'blocked' -ErrorCode 'request_root_not_allowed' -ExitCode 2
}
$audioPath = [IO.Path]::GetFullPath($Audio)
if (-not (Test-Path -LiteralPath $audioPath -PathType Leaf)) {
    Write-BoundedReceipt -Status 'blocked' -ErrorCode 'audio_file_missing' -ExitCode 2
}
if (-not (Test-Path -LiteralPath $brokerPath -PathType Leaf)) {
    Write-BoundedReceipt -Status 'blocked' -ErrorCode 'secret_broker_unavailable' -ExitCode 2
}

$null = New-Item -ItemType Directory -Path $resolvedRequestRoot -Force
$pending = @(Get-ChildItem -LiteralPath $resolvedRequestRoot -Filter '*.pending.json' -File)
if ($pending.Count -ne 0) {
    Write-BoundedReceipt -Status 'blocked' -ErrorCode 'pending_request_ambiguous' -ExitCode 2
}

$jobId = [Guid]::NewGuid().ToString()
$requestPath = Join-Path $resolvedRequestRoot ($jobId + '.pending.json')
$resultPath = Join-Path $resolvedRequestRoot ($jobId + '.result.json')
$request = [ordered]@{
    schema = 'chineseasr.qwen-audio3-important-request.v1'
    job_id = $jobId
    importance = 'important'
    cloud_upload_authorized = $true
    audio_path = $audioPath
    created_utc = [DateTimeOffset]::UtcNow.ToString('o')
    chunk_sec = $ChunkSec
    overlap_sec = $OverlapSec
}
[IO.File]::WriteAllText(
    $requestPath,
    ($request | ConvertTo-Json -Depth 8),
    $utf8NoBom
)

$brokerOutput = ''
$brokerExitCode = 1
try {
    $brokerOutput = & $brokerPath `
        -Action AgentSecretRef `
        -Query $brokerTarget `
        -RuntimePrincipal Codex `
        -Json 2>&1 | Out-String
    $brokerExitCode = $LASTEXITCODE
}
finally {
    if (Test-Path -LiteralPath $requestPath -PathType Leaf) {
        $unclaimedPath = Join-Path $resolvedRequestRoot ($jobId + '.unclaimed.json')
        Move-Item -LiteralPath $requestPath -Destination $unclaimedPath -Force
    }
}

if (-not (Test-Path -LiteralPath $resultPath -PathType Leaf)) {
    $receipt = [ordered]@{
        schema = 'chineseasr.qwen-audio3-important-result.v1'
        job_id = $jobId
        status = 'failed'
        error_code = if ($brokerExitCode -eq 0) {
            'cloud_worker_result_missing'
        }
        else {
            'secret_broker_target_failed'
        }
        important_only = $true
        cloud_upload_performed = $false
        broker_exit_code = $brokerExitCode
        plaintext_returned = $false
        secret_returned = $false
    }
    if ($Json) {
        $receipt | ConvertTo-Json -Depth 8 -Compress | Write-Output
    }
    else {
        $receipt
    }
    exit 3
}

$result = Get-Content -LiteralPath $resultPath -Raw -Encoding utf8 |
    ConvertFrom-Json -Depth 30
if (
    [string]$result.schema -cne 'chineseasr.qwen-audio3-important-result.v1' -or
    [string]$result.job_id -cne $jobId -or
    $result.important_only -ne $true
) {
    Write-BoundedReceipt -Status 'failed' -ErrorCode 'cloud_worker_result_invalid' -ExitCode 3
}

$allowedCredentialResults = @(
    'Invalid',
    'Revoked',
    'Expired',
    'Timeout',
    'Rate-Limited',
    'Provider-5xx',
    'Network-Failure',
    'Scope-Error',
    'Permission-Denied',
    'Provider-Unavailable',
    'Success'
)
$credentialResult = [string]$result.credential_result
$credentialReportStatus = 'not-reported'
if ($credentialResult -cin $allowedCredentialResults) {
    try {
        $reportOutput = & $brokerPath `
            -Action ReportCredentialResult `
            -Query 'qwen-default' `
            -ResultCode $credentialResult `
            -OperationId $jobId `
            -RuntimePrincipal Codex `
            -Json 2>&1 | Out-String
        if ($LASTEXITCODE -eq 0) {
            $credentialReportStatus = 'reported'
        }
        else {
            $credentialReportStatus = 'report-failed'
        }
    }
    catch {
        $credentialReportStatus = 'report-failed'
    }
}

$result | Add-Member -NotePropertyName result_path -NotePropertyValue $resultPath -Force
$result | Add-Member `
    -NotePropertyName credential_report_status `
    -NotePropertyValue $credentialReportStatus `
    -Force
$result | Add-Member -NotePropertyName plaintext_returned -NotePropertyValue $false -Force
$result | Add-Member -NotePropertyName secret_returned -NotePropertyValue $false -Force
if ($Json) {
    $result | ConvertTo-Json -Depth 30 -Compress | Write-Output
}
else {
    $result
}
if ([string]$result.status -ceq 'succeeded') {
    exit 0
}
exit 3
