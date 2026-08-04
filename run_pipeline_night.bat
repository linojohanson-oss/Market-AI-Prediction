@echo off
setlocal
set "PROJECT_DIR=H:\D4uDev\Propios\Market-AI-Prediction"
set "PYTHON_EXE=%PROJECT_DIR%\.venv\Scripts\python.exe"
set "SCRIPT=%PROJECT_DIR%\run_all_markets.py"
:: Create a timestamp like 2026-08-01_16-03-45
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set "dt=%%I"
set "TIMESTAMP=%dt:~0,4%-%dt:~4,2%-%dt:~6,2%_%dt:~8,2%-%dt:~10,2%-%dt:~12,2%"

set "LOG_FILE=%PROJECT_DIR%\logs\pipeline_night_%TIMESTAMP%.log"

cd /d "%PROJECT_DIR%"
if not exist "%PROJECT_DIR%\logs" mkdir "%PROJECT_DIR%\logs"

if not exist "%PYTHON_EXE%" (
    echo Python virtualenv not found: %PYTHON_EXE%
    exit /b 1
)

"%PYTHON_EXE%" "%SCRIPT%" -crypto > "%LOG_FILE%" 2>&1
exit /b %ERRORLEVEL%