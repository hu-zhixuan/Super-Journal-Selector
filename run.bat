@echo off
title Journal Profile Assistant WebUI
echo ========================================================
echo   Journal Profile Assistant (WebUI)
echo   Address: http://127.0.0.1:7860
echo ========================================================

REM Try interpreters in order: python / py launcher / common per-user install path
python --version >nul 2>nul
if %errorlevel%==0 (
    python app.py
    goto :done
)

py -3 --version >nul 2>nul
if %errorlevel%==0 (
    py -3 app.py
    goto :done
)

if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" app.py
    goto :done
)

echo [Error] Python 3.10+ not found. Please install Python and add it to PATH.

:done
pause
