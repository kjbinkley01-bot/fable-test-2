@echo off
rem Puts a "StemDeck" shortcut on the Desktop that starts the app with one click.
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$d = [Environment]::GetFolderPath('Desktop');" ^
  "$s = (New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $d 'StemDeck.lnk'));" ^
  "$s.TargetPath = (Join-Path '%CD%' '.venv\Scripts\pythonw.exe');" ^
  "$s.Arguments = '-m stemdeck app';" ^
  "$s.WorkingDirectory = '%CD%';" ^
  "$s.Description = 'StemDeck';" ^
  "$s.Save()"
if errorlevel 1 (
    echo Could not create the Desktop shortcut. You can still start the app with StemDeck.bat.
) else (
    echo A "StemDeck" shortcut was created on your Desktop.
)
