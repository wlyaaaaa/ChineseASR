[CmdletBinding()]
param(
    [ValidateSet('Install', 'Start', 'Stop', 'Status', 'Uninstall')]
    [string]$Mode = 'Status',
    [switch]$SkipDependencies
)
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$Pythonw = Join-Path $Root '.venv\Scripts\pythonw.exe'
$TaskName = 'ChineseASR Dictation'
$ShortcutName = '中文听写.lnk'
$Launcher = Join-Path $Root 'scripts\Start-Dictation.vbs'
$Icon = Join-Path $Root 'assets\chinese-dictation.ico'
$Wscript = Join-Path $env:WINDIR 'System32\wscript.exe'
if (-not (Test-Path -LiteralPath $Pythonw)) {
    throw 'ChineseASR Python environment is missing. Run setup-core.ps1 and setup-qwen.ps1 first.'
}

function Get-DictationShortcutPath {
    return (Join-Path ([Environment]::GetFolderPath('Programs')) $ShortcutName)
}

function Get-DictationShortcutState {
    $path = Get-DictationShortcutPath
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        return [pscustomobject]@{ path = $path; present = $false; owned = $false; icon_match = $false; target = ''; arguments = ''; icon = '' }
    }
    $shell = New-Object -ComObject WScript.Shell
    $link = $shell.CreateShortcut($path)
    $target = [string]$link.TargetPath
    $arguments = [string]$link.Arguments
    $icon = [string]$link.IconLocation
    $expectedArguments = '"' + $Launcher + '"'
    $expectedIcon = [IO.Path]::GetFullPath($Icon) + ',0'
    $targetFull = $target
    if (-not [string]::IsNullOrWhiteSpace($target)) {
        try { $targetFull = [IO.Path]::GetFullPath($target) } catch { $targetFull = $target }
    }
    $owned = $targetFull -ieq [IO.Path]::GetFullPath($Wscript) -and $arguments -ceq $expectedArguments
    return [pscustomobject]@{ path = $path; present = $true; owned = $owned; icon_match = ($icon -ieq $expectedIcon); target = $target; arguments = $arguments; icon = $icon }
}

function Install-DictationShortcut {
    if (-not (Test-Path -LiteralPath $Launcher -PathType Leaf)) { throw 'ChineseASR start launcher is missing.' }
    if (-not (Test-Path -LiteralPath $Icon -PathType Leaf)) { throw 'ChineseASR shortcut icon is missing.' }
    if (-not (Test-Path -LiteralPath $Wscript -PathType Leaf)) { throw 'Windows Script Host is missing.' }
    $state = Get-DictationShortcutState
    if ($state.present -and -not $state.owned) { throw 'The ChineseASR Start Menu shortcut path is owned by another target.' }
    $shell = New-Object -ComObject WScript.Shell
    $link = $shell.CreateShortcut($state.path)
    $link.TargetPath = $Wscript
    $link.Arguments = '"' + $Launcher + '"'
    $link.IconLocation = [IO.Path]::GetFullPath($Icon) + ',0'
    $link.WorkingDirectory = $Root
    $link.Description = 'Start local ChineseASR Win+H dictation.'
    $link.Save()
    $after = Get-DictationShortcutState
    if (-not $after.owned -or -not $after.icon_match) { throw 'ChineseASR Start Menu shortcut readback failed.' }
}

function Remove-DictationShortcut {
    $state = Get-DictationShortcutState
    if ($state.present -and $state.owned) {
        Remove-Item -LiteralPath $state.path -Force
    }
}

function Test-DictationRunning {
    $result = & $Python -c 'from zh_asr.dictation_windows import is_running; print(int(is_running()))'
    if ($LASTEXITCODE -ne 0) { throw 'Could not inspect the dictation process.' }
    return (($result | Select-Object -Last 1) -eq '1')
}

function Request-DictationStart {
    & $Python -m zh_asr.dictation --start
    return ($LASTEXITCODE -eq 0)
}

function Start-DictationHost {
    # A running host owns the single instance.  Signal its existing UI instead
    # of relying on Task Scheduler's IgnoreNew policy to discard this launch.
    if (Request-DictationStart) { return }

    Start-ScheduledTask -TaskName $TaskName -TaskPath '\'
    $deadline = [DateTime]::UtcNow.AddSeconds(30)
    do {
        if ((Test-DictationRunning) -and (Request-DictationStart)) { return }
        if ([DateTime]::UtcNow -ge $deadline) {
            throw 'Dictation host did not expose its start command after the scheduled task launch.'
        }
        Start-Sleep -Milliseconds 250
    } while ($true)
}

function Stop-DictationGracefully {
    & $Python -m zh_asr.dictation --stop
    if ($LASTEXITCODE -ne 0) { throw 'Could not request dictation shutdown.' }
    $deadline = [DateTime]::UtcNow.AddSeconds(40)
    while (Test-DictationRunning) {
        if ([DateTime]::UtcNow -ge $deadline) {
            throw 'Dictation is still finishing. Retry after the current inference completes.'
        }
        Start-Sleep -Milliseconds 300
    }
    # Wait for Task Scheduler to observe exit before a subsequent Start.
    # Otherwise IgnoreNew can swallow an immediate Stop/Start pair.
    do {
        $CurrentTask = Get-ScheduledTask -TaskName $TaskName -TaskPath '\' -ErrorAction SilentlyContinue
        if (-not $CurrentTask -or $CurrentTask.State -ne 'Running') { break }
        if ([DateTime]::UtcNow -ge $deadline) { throw 'Task Scheduler has not observed dictation exit yet.' }
        Start-Sleep -Milliseconds 300
    } while ($true)
}

Push-Location $Root
try {
    if ($Mode -eq 'Install') {
        if (-not $SkipDependencies) {
            $DownloadRoot = 'E:\Downloads\ChineseASR\dictation'
            $TempRoot = 'E:\Cache\Codex\Temp\chineseasr-dictation-install'
            New-Item -ItemType Directory -Force -Path $DownloadRoot, $TempRoot | Out-Null
            $env:TEMP = $TempRoot
            $env:TMP = $TempRoot
            $env:TMPDIR = $TempRoot
            & $Python -m pip download -r (Join-Path $Root 'requirements-dictation.txt') --dest $DownloadRoot --index-url https://pypi.org/simple
            if ($LASTEXITCODE -ne 0) { throw 'Could not download the dictation dependencies.' }
            & $Python -m pip install --no-index --find-links $DownloadRoot -r (Join-Path $Root 'requirements-dictation.txt')
            if ($LASTEXITCODE -ne 0) { throw 'Could not install the dictation dependencies.' }
        }
        & $Python -m pip check
        if ($LASTEXITCODE -ne 0) { throw 'Python dependencies are inconsistent.' }
        & $Python -c 'import sounddevice, pystray, tkinter; from PIL import Image, ImageDraw; from zh_asr.dictation import DictationSettings; from zh_asr import dictation_windows; DictationSettings.load()'
        if ($LASTEXITCODE -ne 0) { throw 'Dictation prerequisites are unavailable.' }
        $User = [Security.Principal.WindowsIdentity]::GetCurrent().Name
        $Action = New-ScheduledTaskAction -Execute $Pythonw -Argument '-m zh_asr.dictation' -WorkingDirectory $Root
        $Trigger = New-ScheduledTaskTrigger -AtLogOn -User $User
        $Principal = New-ScheduledTaskPrincipal -UserId $User -LogonType Interactive -RunLevel Limited
        $Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
        Register-ScheduledTask -TaskName $TaskName -TaskPath '\' -Action $Action -Trigger $Trigger -Principal $Principal -Settings $Settings -Description 'Local ChineseASR Win+H dictation for the signed-in user. Restore with scripts\dictation.ps1 -Mode Install.' -Force | Out-Null
        Install-DictationShortcut
        Start-ScheduledTask -TaskName $TaskName -TaskPath '\'
    } elseif ($Mode -eq 'Start') {
        Start-DictationHost
    } elseif ($Mode -eq 'Stop') {
        Stop-DictationGracefully
    } elseif ($Mode -eq 'Uninstall') {
        Stop-DictationGracefully
        $Existing = Get-ScheduledTask -TaskName $TaskName -TaskPath '\' -ErrorAction SilentlyContinue
        if ($Existing) {
            Unregister-ScheduledTask -TaskName $TaskName -TaskPath '\' -Confirm:$false
        }
        Remove-DictationShortcut
    }
    $Task = Get-ScheduledTask -TaskName $TaskName -TaskPath '\' -ErrorAction SilentlyContinue
    $Shortcut = Get-DictationShortcutState
    [pscustomobject]@{
        installed = [bool]$Task
        running = Test-DictationRunning
        task_name = $TaskName
        task_state = if ($Task) { [string]$Task.State } else { 'NotInstalled' }
        hotkey = 'Win+H'
        hotkeys = @('Win+H', 'Ctrl+Win+H')
        config = (Join-Path $Root 'configs\dictation.yaml')
        shortcut_path = $Shortcut.path
        shortcut_present = $Shortcut.present
        shortcut_owned = $Shortcut.owned
        shortcut_icon = $Shortcut.icon
        shortcut_icon_match = $Shortcut.icon_match
    } | ConvertTo-Json
} finally {
    Pop-Location
}
