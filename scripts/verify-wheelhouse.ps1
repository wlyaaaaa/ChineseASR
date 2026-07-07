param(
  [string]$Wheelhouse = '',
  [string]$ChecksumFile = ''
)

$ErrorActionPreference = 'Stop'

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $Wheelhouse) {
  $Wheelhouse = Join-Path $Root 'offline\wheelhouse'
}
if (-not $ChecksumFile) {
  $ChecksumFile = Join-Path $Root 'offline\manifests\wheelhouse.sha256'
}

if (-not (Test-Path $Wheelhouse)) {
  throw "Wheelhouse not found: $Wheelhouse"
}
if (-not (Test-Path $ChecksumFile)) {
  throw "Checksum file not found: $ChecksumFile"
}

$Checked = 0
foreach ($Line in Get-Content $ChecksumFile) {
  $Clean = $Line.Trim()
  if (-not $Clean -or $Clean.StartsWith('#')) {
    continue
  }
  if ($Clean -notmatch '^([a-fA-F0-9]{64})\s+\*?(.+)$') {
    throw "Invalid checksum line: $Line"
  }

  $Expected = $Matches[1].ToLowerInvariant()
  $Name = $Matches[2].Trim()
  $Path = Join-Path $Wheelhouse $Name
  if (-not (Test-Path $Path)) {
    throw "Missing wheelhouse file: $Name"
  }

  $Actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
  if ($Actual -ne $Expected) {
    throw "Checksum mismatch: $Name"
  }
  $Checked += 1
}

if ($Checked -eq 0) {
  throw "No checksums found in: $ChecksumFile"
}

Write-Host "Verified $Checked wheelhouse files."
