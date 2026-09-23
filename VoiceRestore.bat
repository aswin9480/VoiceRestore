@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
cd /d "%~dp0"
title VoiceRestore Studio
:menu
cls
echo ========================================================
echo                  VOICERESTORE STUDIO
echo           AI Speech Isolation ^& Restoration
echo ========================================================
echo.
echo  [1] Setup / Repair Environment
echo  [2] Run VoiceRestore Studio
echo  [3] Close
echo.
echo ========================================================
choice /c 123 /n /m " Select an option: "
if errorlevel 3 goto close
if errorlevel 2 goto run
if errorlevel 1 goto setup
:setup
set "VRPY="
py -3.11 -c "import sys" >nul 2>&1
if not errorlevel 1 set "VRPY=py -3.11"
if defined VRPY goto install
py -3.10 -c "import sys" >nul 2>&1
if not errorlevel 1 set "VRPY=py -3.10"
if defined VRPY goto install
python -c "import sys; sys.exit(0 if sys.version_info[:2] in [(3,10),(3,11)] else 1)" >nul 2>&1
if not errorlevel 1 set "VRPY=python"
if defined VRPY goto install
echo [ERROR] Install 64-bit Python 3.10 or 3.11 from python.org and enable Add to PATH.
pause
goto menu
:install
%VRPY% scripts\setup_env.py
if errorlevel 1 echo [WARN] Setup needs attention. Read logs\setup.log for details.
pause
goto menu
:run
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Environment missing. Select Setup first.
    pause
    goto menu
)
call ".venv\Scripts\activate.bat"
python scripts\health_check.py
if errorlevel 1 (
    pause
    goto menu
)
echo [INFO] Starting localhost UI. Press Ctrl+C here to stop it.
python app.py
pause
goto menu
:close
endlocal
exit /b 0
