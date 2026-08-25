Write-Host "register_morning_task.ps1 está deprecado. Reenviando a register_evening_task.ps1..."

$eveningScript = "H:\D4uDev\Propios\Market-AI-Prediction\register_evening_task.ps1"
if (-not (Test-Path $eveningScript)) {
    throw "No se encontró el script de registro evening: $eveningScript"
}

powershell -ExecutionPolicy Bypass -File $eveningScript
