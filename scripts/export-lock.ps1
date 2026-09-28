param(
  [string]$RuntimeRoot = 'E:\Projects\Tools\ChineseASR',
  [Parameter(Mandatory = $true)][string]$Bundle,
  [string]$Distro = 'Ubuntu'
)

$ErrorActionPreference = 'Stop'
$Python = Join-Path $RuntimeRoot '.venv\Scripts\python.exe'
$Config = Join-Path $RuntimeRoot 'configs\models.yaml'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf) -or
    -not (Test-Path -LiteralPath $Config -PathType Leaf)) {
  throw 'Installed ChineseASR Python or model config is missing.'
}
$ManifestDir = Join-Path $Bundle 'manifests'
New-Item -ItemType Directory -Force -Path $ManifestDir | Out-Null
$WindowsLock = Join-Path $ManifestDir 'windows-requirements.txt'
$LinuxLock = Join-Path $ManifestDir 'linux-requirements.txt'

# pip list expresses local/direct installs as their installed pinned versions.
# The project source is restored from Git, not as a third-party wheel.
$WindowsPackages = @(& $Python -m pip list --format=freeze | Where-Object {
  $_ -match '^[A-Za-z0-9_.-]+==[^\s]+$' -and $_ -notmatch '^local-chinese-asr=='
} | Sort-Object)
if ($LASTEXITCODE -ne 0 -or $WindowsPackages.Count -lt 20) {
  throw 'Cannot export installed Windows dependency lock.'
}
$WindowsPackages = @($WindowsPackages + 'wheel==0.47.0' | Sort-Object -Unique)
$WindowsPackages | Set-Content -LiteralPath $WindowsLock -Encoding utf8NoBOM

$WslPython = (& $Python -c "import sys,yaml; print(yaml.safe_load(open(sys.argv[1],encoding='utf-8'))['engines']['fireredasr2-llm']['options']['python_path'])" $Config | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or -not $WslPython.StartsWith('/')) {
  throw 'Invalid configured FireRed WSL Python path.'
}
$LinuxPackages = @(& wsl.exe -d $Distro -- $WslPython -m pip list --format=freeze | Where-Object {
  $_ -match '^[A-Za-z0-9_.-]+==[^\s]+$'
} | Sort-Object)
if ($LASTEXITCODE -ne 0 -or $LinuxPackages.Count -lt 20) {
  throw 'Cannot export installed FireRed WSL dependency lock.'
}
$LinuxPackages | Set-Content -LiteralPath $LinuxLock -Encoding utf8NoBOM

$Versions = [ordered]@{
  schema = 'zh_asr.offline_runtime_versions.v1'
  windows_python = ((& $Python --version | Out-String).Trim())
  linux_python = ((& wsl.exe -d $Distro -- $WslPython --version | Out-String).Trim())
  wsl_distribution = $Distro
  wsl_python = $WslPython
  model_config_sha256 = (Get-FileHash -LiteralPath $Config -Algorithm SHA256).Hash.ToLowerInvariant()
}
$Versions | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $ManifestDir 'versions.json') -Encoding utf8NoBOM
Write-Host "Windows lock: $($WindowsPackages.Count) packages; FireRed WSL lock: $($LinuxPackages.Count) packages."
