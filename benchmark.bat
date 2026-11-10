@echo off
rem Measures stem separation speed and reports whether the GPU was used.
rem   benchmark.bat                    4:00 generated test signal
rem   benchmark.bat "C:\Music\a.mp3"   a real song (drag a file onto this .bat)
rem   benchmark.bat "a.mp3" --sweep    try several settings and save the fastest
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run setup.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" scripts\benchmark.py %*
echo.
pause
