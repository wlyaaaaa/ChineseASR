param(
  [string]$Wheelhouse = '',
  [string]$ManifestDir = '',
  [string]$TorchIndexUrl = 'https://download.pytorch.org/whl/cu128',
  [string]$PypiIndexUrl = 'https://pypi.tuna.tsinghua.edu.cn/simple',
  [string]$PypiTrustedHost = 'pypi.tuna.tsinghua.edu.cn'
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Invoke-NoProxy.ps1')
Clear-ProxyEnv

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $Wheelhouse) {
  $Wheelhouse = Join-Path $Root 'offline\wheelhouse'
}
if (-not $ManifestDir) {
  $ManifestDir = Join-Path $Root 'offline\manifests'
}

$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) {
  throw 'Virtual environment not found. Run scripts\install-torch-cu128-direct.ps1 and scripts\setup-core.ps1 first.'
}

$LockFile = Join-Path $ManifestDir 'requirements-lock.txt'
if (-not (Test-Path $LockFile)) {
  throw "Lock file not found: $LockFile. Run scripts\export-lock.ps1 first."
}

New-Item -ItemType Directory -Force -Path $Wheelhouse | Out-Null
New-Item -ItemType Directory -Force -Path $ManifestDir | Out-Null

$TempDir = Join-Path $ManifestDir '_tmp'
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null
$TorchRequirements = Join-Path $TempDir 'requirements-torch.txt'
$OtherRequirements = Join-Path $TempDir 'requirements-other.txt'

$TorchLines = New-Object System.Collections.Generic.List[string]
$OtherLines = New-Object System.Collections.Generic.List[string]
foreach ($Line in Get-Content $LockFile) {
  $Clean = $Line.Trim()
  if (-not $Clean -or $Clean.StartsWith('#')) {
    continue
  }
  if ($Clean -match '^(torch|torchaudio|torchvision)(==|~=|>=|<=|>|<|$)') {
    $TorchLines.Add($Clean)
  } else {
    $OtherLines.Add($Clean)
  }
}

if ($TorchLines.Count -gt 0) {
  $TorchLines | Set-Content -Encoding UTF8 $TorchRequirements
  & $Python -m pip download -r $TorchRequirements -d $Wheelhouse --index-url $TorchIndexUrl --extra-index-url $PypiIndexUrl --trusted-host download.pytorch.org --trusted-host $PypiTrustedHost
  if ($LASTEXITCODE -ne 0) {
    throw 'pip download failed for torch requirements.'
  }
}

if ($OtherLines.Count -gt 0) {
  $OtherLines | Set-Content -Encoding UTF8 $OtherRequirements
  & $Python -m pip download -r $OtherRequirements -d $Wheelhouse -i $PypiIndexUrl --trusted-host $PypiTrustedHost
  if ($LASTEXITCODE -ne 0) {
    throw 'pip download failed for non-torch requirements.'
  }
}

$ChecksumFile = Join-Path $ManifestDir 'wheelhouse.sha256'
$JsonFile = Join-Path $ManifestDir 'wheelhouse.json'
$Files = Get-ChildItem -LiteralPath $Wheelhouse -File | Sort-Object Name
if (-not $Files) {
  throw "Wheelhouse is empty: $Wheelhouse"
}

$ChecksumLines = foreach ($File in $Files) {
  $Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $File.FullName).Hash.ToLowerInvariant()
  "$Hash  $($File.Name)"
}
$ChecksumLines | Set-Content -Encoding ASCII $ChecksumFile

$LockHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $LockFile).Hash.ToLowerInvariant()
$Manifest = [ordered]@{
  schema_version = 1
  generated_at = (Get-Date).ToString('s')
  wheelhouse = $Wheelhouse
  requirements_lock = $LockFile
  requirements_lock_sha256 = $LockHash
  files = @(
    foreach ($File in $Files) {
      [ordered]@{
        name = $File.Name
        size_bytes = $File.Length
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $File.FullName).Hash.ToLowerInvariant()
      }
    }
  )
}
$Manifest | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $JsonFile

Remove-Item -LiteralPath $TempDir -Recurse -Force
Write-Host "Wheelhouse: $Wheelhouse"
Write-Host "Checksum file: $ChecksumFile"
Write-Host "Manifest: $JsonFile"
