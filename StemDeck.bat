@echo off
rem Starts StemDeck. Double-click, or drag a song onto this file.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo StemDeck is not set up yet. Run setup.bat first.
    pause
    exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" -m stemdeck app %*
