@echo off
rem ==========================================================================
rem  StemDeck setup. Run once (double-click, or "setup.bat" in a terminal).
rem  Safe to run again: it only installs what is missing.
rem
rem    setup.bat          install with AMD GPU support (ROCm build of PyTorch)
rem    setup.bat cpu      install CPU-only (no GPU acceleration; for testing)
rem ==========================================================================
setlocal
cd /d "%~dp0"

set "GPU_ARCH=gfx1201"
if not "%STEMDECK_GPU_ARCH%"=="" set "GPU_ARCH=%STEMDECK_GPU_ARCH%"
set "ROCM_INDEX=https://stable.repo.amd.com/rocm/whl-next/"
set "MODE=gpu"
if /i "%~1"=="cpu" set "MODE=cpu"

echo.
echo === StemDeck setup (%MODE%) ===
echo.

rem ---- 1. Python 3.12 ------------------------------------------------------
py -3.12 -c "import sys" >nul 2>&1
if errorlevel 1 (
    echo Python 3.12 was not found.
    echo Install it by running this command, then run setup.bat again:
    echo.
    echo     winget install -e --id Python.Python.3.12
    echo.
    pause
    exit /b 1
)

rem ---- 2. Private environment in .venv (nothing is installed system-wide) ---
if not exist ".venv\Scripts\python.exe" (
    echo [1/5] Creating Python environment in .venv ...
    py -3.12 -m venv .venv
    if errorlevel 1 (
        echo Could not create the Python environment.
        pause
        exit /b 1
    )
) else (
    echo [1/5] Python environment already exists.
)
set "VPY=.venv\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip --quiet

rem ---- 3. PyTorch ----------------------------------------------------------
if "%MODE%"=="cpu" (
    echo [2/5] Installing CPU-only PyTorch ...
    "%VPY%" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
) else (
    echo [2/5] Installing AMD ROCm PyTorch for %GPU_ARCH% - this is a multi-GB download ...
    "%VPY%" -m pip install --index-url %ROCM_INDEX% "torch[device-%GPU_ARCH%]==2.14.0" "torchaudio==2.11.*"
)
if errorlevel 1 (
    echo.
    echo PyTorch installation failed. Copy the messages above and send them back.
    pause
    exit /b 1
)

rem ---- 4. Everything else, without letting pip swap out the GPU PyTorch -----
echo [3/5] Installing the remaining packages ...
"%VPY%" -m pip freeze | findstr /b /i "torch" > ".venv\torch-constraints.txt"
"%VPY%" -m pip install -r requirements.txt -c ".venv\torch-constraints.txt"
if errorlevel 1 (
    echo.
    echo Package installation failed. Copy the messages above and send them back.
    pause
    exit /b 1
)

rem ---- 5. Download the AI models so the first song does not wait for them ---
echo [4/5] Downloading models (about 700 MB, one time) ...
"%VPY%" -m stemdeck.fetch_models
if errorlevel 1 (
    echo Model download failed - they will be downloaded on first use instead.
)

rem ---- 6. Report what separation will run on --------------------------------
echo [5/5] Checking the GPU ...
echo.
"%VPY%" -m stemdeck devices
echo.
echo Setup finished.
echo   Next: run  benchmark.bat  to measure separation speed,
echo   then double-click  StemDeck.bat  to start the app.
echo.
pause
