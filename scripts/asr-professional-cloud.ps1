#requires -Version 7.2

[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string] $Audio,

    [switch] $Important,

    [switch] $QualityReview,

    [switch] $CloudUploadAuthorized,

    [switch] $AutomaticReview,

    [switch] $CloudFirst,

    [string] $ResumeJobId = '',

    [string] $ModelProfile = '',

    [string] $LocalOutDir = '',

    [ValidateSet('', 'verified', 'provisional', 'unavailable', 'pending', 'not_applicable')]
    [string] $EvidenceStatus = '',

    [hashtable] $Hotwords = @{},

    [switch] $SpeakerDiarization,

    [switch] $KeepDialect,

    [Nullable[int]] $ChannelIndex = $null,

    [ValidateRange(1, 180)]
    [int] $ChunkSec = 180,

    [ValidateRange(0, 179)]
    [int] $OverlapSec = 1,

    [string] $RequestRoot = '',

    [switch] $PersonalDataGuard,

    [string] $ProgressPath = '',

    [switch] $Json
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$utf8NoBom = [Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8NoBom
$OutputEncoding = $utf8NoBom

$brokerPath = 'C:\ProgramData\PCConfig\AuthorityHost\tools\Invoke-SecretBroker.ps1'
$pipelineScript = Join-Path $PSScriptRoot 'cloud-review-pipeline.py'
$python = Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { $python = 'python' }
$importantResultSchema = 'chineseasr.qwen-audio3-important-result.v1'
$qualityReviewResultSchema = 'chineseasr.qwen-audio3-quality-review-result.v1'
$transcriptionResultSchema = 'chineseasr.qwen-audio3-transcription-result.v1'
$canonicalRequestRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\outputs\cloud-jobs'))
if ([string]::IsNullOrWhiteSpace($RequestRoot)) { $RequestRoot = $canonicalRequestRoot }

function Write-BoundedReceipt {
    param(
        [Parameter(Mandatory)][string] $Status,
        [Parameter(Mandatory)][string] $ErrorCode,
        [Parameter(Mandatory)][int] $ExitCode,
        [string] $Message = '',
        [string] $BillingWarning = ''
    )

    $receipt = [ordered]@{
        schema = $selectedResultSchema
        status = $Status
        error_code = $ErrorCode
        purpose = $selectedPurpose
        important_only = $selectedImportantOnly
        cloud_upload_performed = $false
        plaintext_returned = $false
        secret_returned = $false
        cloud_retry_policy = 'do_not_retry'
        local_fallback_recommendation = 'none'
    }
    if ($Message) { $receipt['message'] = $Message }
    if ($BillingWarning) { $receipt['billing_warning'] = $BillingWarning }
    if ($Json) {
        $receipt | ConvertTo-Json -Depth 8 -Compress | Write-Output
    }
    else {
        $receipt
    }
    exit $ExitCode
}

function Get-SafeBrokerErrorCode {
    param([AllowEmptyString()][string] $Raw)
    if ([string]::IsNullOrWhiteSpace($Raw)) { return $null }
    try { $value = $Raw | ConvertFrom-Json -Depth 20 }
    catch { return $null }
    if (
        [string]$value.schema -cne 'pcconfig.secret-broker-result.v1' -or
        [string]$value.status -cne 'error'
    ) { return $null }
    $code = [string]$value.error
    if ($code -cmatch '^[a-z][a-z0-9_:-]{0,95}$') { return $code }
    return $null
}

function Convert-BrokerReceiptForLog {
    param([AllowEmptyString()][string] $Raw)
    if ([string]::IsNullOrWhiteSpace($Raw)) { return $null }
    try { return ($Raw | ConvertFrom-Json -AsHashtable -Depth 30) }
    catch { return $Raw.Trim() }
}

function Write-BrokerReceiptLog {
    param(
        [string] $Path,
        [string] $JobId,
        [AllowEmptyString()][string] $AgentSecretRefRaw,
        [AllowEmptyString()][string] $CredentialReportRaw
    )
    $log = [ordered]@{
        schema = 'zh_asr.broker_receipts.v1'
        job_id = $JobId
        agent_secret_ref_receipt = Convert-BrokerReceiptForLog -Raw $AgentSecretRefRaw
        credential_result_report_receipt = Convert-BrokerReceiptForLog -Raw $CredentialReportRaw
    }
    [IO.File]::WriteAllText($Path, ($log | ConvertTo-Json -Depth 30), $utf8NoBom)
}

function Save-PreviousCloudAttempt {
    param([string] $Root, [string] $JobId)
    $existing = @('provider', 'broker', 'result') | ForEach-Object {
        $path = Join-Path $Root ($JobId + '.' + $_ + '.json')
        if (Test-Path -LiteralPath $path -PathType Leaf) { $path }
    }
    if (-not $existing) { return '' }
    $attemptId = [Guid]::NewGuid().ToString()
    $directory = Join-Path $Root ($JobId + '.attempts/' + $attemptId)
    $null = New-Item -ItemType Directory -Path $directory -Force
    $receipts = @(foreach ($path in $existing) {
        $snapshot = Join-Path $directory ([IO.Path]::GetFileName($path))
        Copy-Item -LiteralPath $path -Destination $snapshot
        [ordered]@{
            source_path = $path
            path = $snapshot
            sha256 = (Get-FileHash -LiteralPath $snapshot -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    })
    $indexPath = Join-Path $directory 'index.json'
    $index = [ordered]@{
        schema = 'zh_asr.cloud_attempt_receipts.v1'; job_id = $JobId
        attempt_id = $attemptId; archived_utc = [DateTimeOffset]::UtcNow.ToString('o')
        receipts = $receipts
    }
    [IO.File]::WriteAllText($indexPath, ($index | ConvertTo-Json -Depth 8), $utf8NoBom)
    return $indexPath
}

function Test-CredentialRecentlyHealthy {
    param([string] $Path, [string] $CredentialRef)
    try {
        $health = Get-Content -LiteralPath $Path -Raw -Encoding utf8 -ErrorAction Stop | ConvertFrom-Json
        $age = [DateTimeOffset]::UtcNow - [DateTimeOffset]::Parse([string]$health.verified_utc)
        return [string]$health.credential_ref -ceq $CredentialRef -and
            $age -ge [TimeSpan]::Zero -and $age -lt [TimeSpan]::FromHours(24)
    }
    catch { return $false }
}

function Save-CredentialHealth {
    param([string] $Path, [string] $CredentialRef, [AllowEmptyString()][string] $ReportRaw)
    try { $receipt = $ReportRaw | ConvertFrom-Json -Depth 20 }
    catch { return }
    if ([string]$receipt.status -cne 'pass' -or $receipt.credential_update_required -ne $false) { return }
    $temporary = $Path + '.' + [Guid]::NewGuid().ToString('N') + '.tmp'
    $health = [ordered]@{ credential_ref = $CredentialRef; verified_utc = [DateTimeOffset]::UtcNow.ToString('o') }
    [IO.File]::WriteAllText($temporary, ($health | ConvertTo-Json), $utf8NoBom)
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

function Get-CloudFailureAdvice {
    param(
        [AllowEmptyString()][string] $Status,
        [AllowEmptyString()][string] $ErrorCode,
        [AllowEmptyString()][string] $CredentialResult
    )
    if ($Status -ceq 'succeeded') {
        return [ordered]@{ cloud_retry_policy = 'not_needed'; local_fallback_recommendation = 'none' }
    }
    if ($ErrorCode -ceq 'runtime_rebind_required') {
        return [ordered]@{
            cloud_retry_policy = 'retry_once_after_runtime_rebind'
            local_fallback_recommendation = 'use_asr_smart_local_until_rebound'
        }
    }
    if ($CredentialResult -cin @('Timeout', 'Rate-Limited', 'Provider-5xx', 'Network-Failure')) {
        return [ordered]@{
            cloud_retry_policy = 'retry_cloud_once_if_still_authorized'
            local_fallback_recommendation = 'use_asr_smart_local'
        }
    }
    return [ordered]@{
        cloud_retry_policy = 'do_not_retry_cloud'
        local_fallback_recommendation = 'use_asr_smart_local'
    }
}

$selectedPurpose = 'unspecified'
$selectedResultSchema = $importantResultSchema
$selectedImportantOnly = $false
if ($Important -and -not $QualityReview) {
    $selectedPurpose = 'important_evidence'
    $selectedResultSchema = $importantResultSchema
    $selectedImportantOnly = $true
}
elseif ($QualityReview -and -not $Important) {
    $selectedPurpose = 'quality_review'
    $selectedResultSchema = $qualityReviewResultSchema
}
elseif ($CloudFirst -and -not $Important) {
    $selectedPurpose = 'transcription'
    $selectedResultSchema = $transcriptionResultSchema
}

if ($CloudFirst -and $QualityReview) {
    Write-BoundedReceipt -Status 'blocked' -ErrorCode 'cloud_use_purpose_ambiguous' -ExitCode 2
}
if ($CloudFirst) { $AutomaticReview = $true }

if ($Important -and $QualityReview) {
    Write-BoundedReceipt -Status 'blocked' -ErrorCode 'cloud_use_purpose_ambiguous' -ExitCode 2
}
if (-not $Important -and -not $QualityReview -and -not $CloudFirst) {
    Write-BoundedReceipt -Status 'blocked' -ErrorCode 'cloud_use_purpose_required' -ExitCode 2
}
if (-not $CloudUploadAuthorized -and -not $AutomaticReview) {
    Write-BoundedReceipt `
        -Status 'blocked' `
        -ErrorCode 'cloud_upload_authorization_required' `
        -ExitCode 2
}
if ($AutomaticReview -and -not $CloudFirst -and -not $LocalOutDir) {
    Write-BoundedReceipt -Status 'blocked' -ErrorCode 'local_review_required' -ExitCode 2
}
if ($OverlapSec -ge $ChunkSec) {
    Write-BoundedReceipt -Status 'blocked' -ErrorCode 'chunk_policy_invalid' -ExitCode 2
}

$resolvedRequestRoot = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($RequestRoot)
if ($resolvedRequestRoot -cne $canonicalRequestRoot) {
    Write-BoundedReceipt -Status 'blocked' -ErrorCode 'request_root_not_allowed' -ExitCode 2
}
$audioPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Audio)
if (-not (Test-Path -LiteralPath $audioPath -PathType Leaf)) {
    Write-BoundedReceipt -Status 'blocked' -ErrorCode 'audio_file_missing' -ExitCode 2
}
# Per audio content: identical bytes resume one job and must take turns even under
# two paths, while different voice messages no longer queue behind each other.
$audioContentHash = (Get-FileHash -LiteralPath $audioPath -Algorithm SHA256).Hash
$cloudMutex = [Threading.Mutex]::new($false, 'Global\ChineseASRCloudUploadOnce-' + $audioContentHash)
$mutexHeld = $false
try {
    try { $mutexHeld = $cloudMutex.WaitOne([TimeSpan]::FromSeconds(5)) }
    catch [Threading.AbandonedMutexException] { $mutexHeld = $true }
}
catch {
    $cloudMutex.Dispose()
    Write-BoundedReceipt -Status 'blocked' -ErrorCode 'cloud_upload_busy' -ExitCode 2
}
if (-not $mutexHeld) {
    $cloudMutex.Dispose()
    Write-BoundedReceipt -Status 'blocked' -ErrorCode 'cloud_upload_busy' -ExitCode 2
}
try {
$null = New-Item -ItemType Directory -Path $resolvedRequestRoot -Force

$jobId = [Guid]::NewGuid().ToString()
$intentPath = Join-Path $resolvedRequestRoot ($jobId + '.intent.json')
$requestPath = Join-Path $resolvedRequestRoot ($jobId + '.pending.json')
$providerPath = Join-Path $resolvedRequestRoot ($jobId + '.provider.json')
$resultPath = Join-Path $resolvedRequestRoot ($jobId + '.result.json')
$brokerLogPath = Join-Path $resolvedRequestRoot ($jobId + '.broker.json')
$intent = [ordered]@{
    schema = 'zh_asr.cloud_review_intent.v1'
    job_id = $jobId
    purpose = $selectedPurpose
    cloud_upload_authorized = $true
    audio_path = $audioPath
    created_utc = [DateTimeOffset]::UtcNow.ToString('o')
    chunk_sec = $ChunkSec
    overlap_sec = $OverlapSec
}
if ($Important) {
    $intent['importance'] = 'important'
}
$intent['automatic_review'] = [bool]$AutomaticReview
if ($CloudFirst) { $intent['cloud_first'] = $true }
$intent['speaker_diarization'] = [bool]$SpeakerDiarization
$intent['keep_dialect'] = [bool]$KeepDialect
$intent['hotwords'] = $Hotwords
if ($null -ne $ChannelIndex) { $intent['channel_index'] = [int]$ChannelIndex }
if ($EvidenceStatus) { $intent['evidence_status'] = $EvidenceStatus }
if ($ModelProfile) { $intent['model_profile'] = $ModelProfile }
if ($LocalOutDir) {
    $localPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($LocalOutDir)
    if (-not (Test-Path -LiteralPath $localPath -PathType Container)) {
        Write-BoundedReceipt -Status 'blocked' -ErrorCode 'local_result_missing' -ExitCode 2
    }
    $intent['local_out_dir'] = $localPath
}
[IO.File]::WriteAllText(
    $intentPath,
    ($intent | ConvertTo-Json -Depth 8),
    $utf8NoBom
)
$prepareArguments = @($pipelineScript, 'prepare', '--root', $resolvedRequestRoot, '--intent', $intentPath)
if ($ResumeJobId) { $prepareArguments += @('--resume-job-id', $ResumeJobId) }
$prepareRaw = & $python @prepareArguments 2>&1 | Out-String
$prepareExit = $LASTEXITCODE
try { $prepared = $prepareRaw | ConvertFrom-Json -Depth 30 }
catch { Write-BoundedReceipt -Status 'failed' -ErrorCode 'cloud_prepare_receipt_invalid' -ExitCode 3 }
if ($prepareExit -ne 0 -or [string]$prepared.status -cne 'ready') {
    $errorCode = if ($prepared.error_code) { [string]$prepared.error_code } else { 'cloud_prepare_failed' }
    $status = if ($prepared.status) { [string]$prepared.status } else { 'failed' }
    $prepareMessage = if ($prepared.PSObject.Properties.Name -contains 'message') { [string]$prepared.message } else { '' }
    $prepareWarning = if ($prepared.PSObject.Properties.Name -contains 'billing_warning') { [string]$prepared.billing_warning } else { '' }
    Write-BoundedReceipt -Status $status -ErrorCode $errorCode -ExitCode 2 `
        -Message $prepareMessage -BillingWarning $prepareWarning
}
$jobId = [string]$prepared.job_id
$intentPath = [string]$prepared.intent_path
$requestPath = [string]$prepared.request_path
$providerPath = Join-Path $resolvedRequestRoot ($jobId + '.provider.json')
$resultPath = Join-Path $resolvedRequestRoot ($jobId + '.result.json')
$brokerLogPath = Join-Path $resolvedRequestRoot ($jobId + '.broker.json')
$brokerTarget = [string]$prepared.secret_ref_target
$credentialRef = [string]$prepared.credential_ref
if (-not $brokerTarget -or -not $credentialRef) {
    Write-BoundedReceipt -Status 'failed' -ErrorCode 'cloud_credential_route_missing' -ExitCode 3
}
$previousAttemptPath = Save-PreviousCloudAttempt -Root $resolvedRequestRoot -JobId $jobId

$brokerOutput = ''
$brokerExitCode = 1
$providerBefore = if (Test-Path -LiteralPath $providerPath -PathType Leaf) {
    (Get-Item -LiteralPath $providerPath).LastWriteTimeUtc.Ticks
} else { 0 }
try {
    if ($PersonalDataGuard) {
        $transmitArguments = @($pipelineScript, 'transmit', '--root', $resolvedRequestRoot,
            '--intent', $intentPath, '--request', $requestPath, '--broker', $brokerPath, '--target', $brokerTarget)
        if ($ProgressPath) { $transmitArguments += @('--progress', $ProgressPath) }
        $brokerOutput = & $python @transmitArguments 2>&1 | Out-String
        $brokerExitCode = $LASTEXITCODE
    }
    elseif (Test-Path -LiteralPath $brokerPath -PathType Leaf) {
        # The broker writes directly to Console.Out; a child process makes that
        # output capturable instead of leaking it into this entry's JSON stdout.
        $brokerOutput = & pwsh -NoProfile -NonInteractive -File $brokerPath `
            -Action AgentSecretRef `
            -Query $brokerTarget `
            -SecretRefRequestPath $requestPath `
            -Json 2>&1 | Out-String
        $brokerExitCode = $LASTEXITCODE
    }
}
finally {
    if (Test-Path -LiteralPath $requestPath -PathType Leaf) {
        $unclaimedPath = Join-Path $resolvedRequestRoot ($jobId + '.unclaimed.json')
        Move-Item -LiteralPath $requestPath -Destination $unclaimedPath -Force
    }
}
Write-BrokerReceiptLog -Path $brokerLogPath -JobId $jobId `
    -AgentSecretRefRaw $brokerOutput -CredentialReportRaw ''

$brokerError = ''
$providerUpdated = (Test-Path -LiteralPath $providerPath -PathType Leaf) -and
    (Get-Item -LiteralPath $providerPath).LastWriteTimeUtc.Ticks -ne $providerBefore
if (-not $providerUpdated) {
    $brokerErrorCode = Get-SafeBrokerErrorCode -Raw $brokerOutput
    $brokerError = if (-not (Test-Path -LiteralPath $brokerPath -PathType Leaf)) {
        'secret_broker_unavailable'
    }
    elseif ($brokerErrorCode) {
        $brokerErrorCode
    }
    elseif ($brokerExitCode -eq 0) {
        'cloud_worker_result_missing'
    }
    else {
        'secret_broker_target_failed'
    }
}
$finalArguments = @($pipelineScript, 'finalize', '--root', $resolvedRequestRoot,
                    '--intent', $intentPath)
if ($brokerError) { $finalArguments += @('--broker-error', $brokerError) }
else { $finalArguments += @('--provider', $providerPath) }
$finalRaw = & $python @finalArguments 2>&1 | Out-String
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $resultPath -PathType Leaf)) {
    Write-BoundedReceipt -Status 'failed' -ErrorCode 'cloud_finalize_failed' -ExitCode 3
}

$result = Get-Content -LiteralPath $resultPath -Raw -Encoding utf8 |
    ConvertFrom-Json -Depth 30
if (
    [string]$result.schema -cne $selectedResultSchema -or
    [string]$result.job_id -cne $jobId -or
    [string]$result.purpose -cne $selectedPurpose -or
    $result.important_only -ne $selectedImportantOnly
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
$reportOutput = ''
$credentialHealthPath = Join-Path $resolvedRequestRoot 'credential-health.json'
if ($credentialResult -cne 'Success') {
    Remove-Item -LiteralPath $credentialHealthPath -Force -ErrorAction SilentlyContinue
}
if ($credentialResult -ceq 'Success' -and (Test-CredentialRecentlyHealthy -Path $credentialHealthPath -CredentialRef $credentialRef)) {
    # A success report only clears update_required; with that already known to be
    # false the broker changes nothing, so the seconds it costs buy nothing.
    $credentialReportStatus = 'skipped_no_state_change'
}
elseif ($credentialResult -cin $allowedCredentialResults) {
    try {
        $reportOutput = & pwsh -NoProfile -NonInteractive -File $brokerPath `
            -Action ReportCredentialResult `
            -Query $credentialRef `
            -ResultCode $credentialResult `
            -OperationId $jobId `
            -Json 2>&1 | Out-String
        if ($LASTEXITCODE -eq 0) {
            $credentialReportStatus = 'reported'
            if ($credentialResult -ceq 'Success') {
                Save-CredentialHealth -Path $credentialHealthPath -CredentialRef $credentialRef -ReportRaw $reportOutput
            }
        }
        else {
            $credentialReportStatus = 'report-failed'
        }
    }
    catch {
        $credentialReportStatus = 'report-failed'
    }
}
Write-BrokerReceiptLog -Path $brokerLogPath -JobId $jobId `
    -AgentSecretRefRaw $brokerOutput -CredentialReportRaw $reportOutput

$result | Add-Member -NotePropertyName result_path -NotePropertyValue $resultPath -Force
$result | Add-Member -NotePropertyName broker_log_path -NotePropertyValue $brokerLogPath -Force
if ($previousAttemptPath) {
    $result | Add-Member -NotePropertyName previous_attempt_path -NotePropertyValue $previousAttemptPath -Force
}
$result | Add-Member `
    -NotePropertyName credential_report_status `
    -NotePropertyValue $credentialReportStatus `
    -Force
$result | Add-Member -NotePropertyName plaintext_returned -NotePropertyValue $false -Force
$result | Add-Member -NotePropertyName secret_returned -NotePropertyValue $false -Force
$advice = Get-CloudFailureAdvice `
    -Status ([string]$result.status) `
    -ErrorCode ([string]$result.error_code) `
    -CredentialResult ([string]$result.credential_result)
$result | Add-Member `
    -NotePropertyName cloud_retry_policy `
    -NotePropertyValue $advice.cloud_retry_policy `
    -Force
$result | Add-Member `
    -NotePropertyName local_fallback_recommendation `
    -NotePropertyValue $advice.local_fallback_recommendation `
    -Force
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
}
finally {
    if ($mutexHeld) { $cloudMutex.ReleaseMutex() }
    $cloudMutex.Dispose()
}
