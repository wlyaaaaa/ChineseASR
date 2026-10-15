$script:ProxyEnvNames = @(
  'HTTP_PROXY',
  'HTTPS_PROXY',
  'ALL_PROXY',
  'http_proxy',
  'https_proxy',
  'all_proxy'
)

$script:NoProxyValue = 'localhost,127.0.0.1,::1,aliyun.com,*.aliyun.com,aliyuncs.com,*.aliyuncs.com,modelscope.cn,*.modelscope.cn'

function Clear-ProxyEnv {
  foreach ($name in $script:ProxyEnvNames) {
    Remove-Item -Path "Env:$name" -ErrorAction SilentlyContinue
  }
  $env:NO_PROXY = $script:NoProxyValue
  $env:no_proxy = $script:NoProxyValue
}

function Invoke-NoProxyCommand {
  param(
    [Parameter(Mandatory = $true)]
    [string]$FilePath,

    [string[]]$ArgumentList = @()
  )

  $PreviousProxyEnvironment = [System.Collections.Generic.Dictionary[string, object]]::new([StringComparer]::Ordinal)
  foreach ($name in ($script:ProxyEnvNames + @('NO_PROXY', 'no_proxy'))) {
    $PreviousProxyEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
  }
  try {
    Clear-ProxyEnv
    & $FilePath @ArgumentList
  } finally {
    foreach ($entry in $PreviousProxyEnvironment.GetEnumerator()) {
      if ($null -eq $entry.Value) {
        Remove-Item -LiteralPath "Env:$($entry.Key)" -ErrorAction SilentlyContinue
      } else {
        [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process')
      }
    }
  }
}

if ($args.Count -gt 0) {
  $cmd = $args[0]
  $rest = @()
  if ($args.Count -gt 1) {
    $rest = $args[1..($args.Count - 1)]
  }
  Invoke-NoProxyCommand -FilePath $cmd -ArgumentList $rest
  exit $LASTEXITCODE
}

