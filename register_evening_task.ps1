$taskName = "MarketAI-Pipeline-Evening"
$scriptPath = "H:\D4uDev\Propios\Market-AI-Prediction\run_pipeline_evening.bat"
$workingDir = "H:\D4uDev\Propios\Market-AI-Prediction"
$runAt = "8:30PM"

if (-not (Test-Path $scriptPath)) {
    throw "No se encontró el script de arranque: $scriptPath"
}

$action = New-ScheduledTaskAction -Execute $scriptPath -WorkingDirectory $workingDir
$trigger = New-ScheduledTaskTrigger -Daily -At $runAt
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null

Get-ScheduledTask -TaskName $taskName | Select-Object TaskName, State, Actions, Triggers | Format-List
