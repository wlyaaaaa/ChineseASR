Option Explicit

Dim shell, fso, scriptFolder, powershellPath, dictationScript, command, exitCode
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

scriptFolder = fso.GetParentFolderName(WScript.ScriptFullName)
dictationScript = fso.BuildPath(scriptFolder, "dictation.ps1")
powershellPath = shell.ExpandEnvironmentStrings("%ProgramFiles%") & "\PowerShell\7\pwsh.exe"

If Not fso.FileExists(powershellPath) Then
    MsgBox "PowerShell 7 was not found.", vbExclamation, "ChineseASR"
    WScript.Quit 2
End If
If Not fso.FileExists(dictationScript) Then
    MsgBox "The ChineseASR start script was not found.", vbExclamation, "ChineseASR"
    WScript.Quit 3
End If

command = Quote(powershellPath) & " -NoLogo -NoProfile -NonInteractive" & _
          " -ExecutionPolicy Bypass -File " & Quote(dictationScript) & " -Mode Start"
exitCode = shell.Run(command, 0, True)
If exitCode <> 0 Then
    MsgBox "ChineseASR could not start (exit code " & CStr(exitCode) & ").", vbExclamation, "ChineseASR"
End If
WScript.Quit exitCode

Function Quote(value)
    Quote = Chr(34) & value & Chr(34)
End Function
