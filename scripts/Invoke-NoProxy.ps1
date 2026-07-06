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

  Clear-ProxyEnv
  & $FilePath @ArgumentList
  return $LASTEXITCODE
}

if ($args.Count -gt 0) {
  Clear-ProxyEnv
  $cmd = $args[0]
  $rest = @()
  if ($args.Count -gt 1) {
    $rest = $args[1..($args.Count - 1)]
  }
  & $cmd @rest
  exit $LASTEXITCODE
}

