@echo off
rem Compares StemDeck's BPM and key with reference values you supply.
rem   check_analysis.bat test_songs.csv
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run setup.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" scripts\check_analysis.py %*
echo.
pause
