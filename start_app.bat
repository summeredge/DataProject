@echo off
setlocal
cd /d "%~dp0"

set "APP_URL=http://127.0.0.1:8765/"
set "PYTHON_CMD="

where python >nul 2>nul
if %errorlevel%==0 set "PYTHON_CMD=python"

if not defined PYTHON_CMD (
  where py >nul 2>nul
  if %errorlevel%==0 set "PYTHON_CMD=py -3"
)

if not defined PYTHON_CMD (
  echo Python was not found.
  echo Please install Python 3.10 or later, then run this file again.
  echo https://www.python.org/downloads/
  pause
  exit /b 1
)

echo Checking Python dependencies...
%PYTHON_CMD% -c "import pandas, numpy" >nul 2>nul
if errorlevel 1 (
  echo Installing required packages...
  %PYTHON_CMD% -m pip install -e .
  if errorlevel 1 (
    echo Failed to install dependencies.
    echo Please check Python, pip, and network settings.
    pause
    exit /b 1
  )
)

echo Checking model explanation dependencies...
%PYTHON_CMD% -c "import sklearn, shap" >nul 2>nul
if errorlevel 1 (
  echo scikit-learn or SHAP was not found. Installing model explanation packages...
  %PYTHON_CMD% -m pip install scikit-learn shap
  if errorlevel 1 (
    echo Failed to install model explanation packages.
    echo Model explanation requires scikit-learn and SHAP. Please check Python, pip, and network settings.
    pause
    exit /b 1
  )
)

echo Starting local web app...
echo Open this URL if the browser does not open automatically:
echo %APP_URL%

start "" powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command "Start-Sleep -Seconds 2; Start-Process '%APP_URL%'"

%PYTHON_CMD% -m chem_ts_corr.cli serve --host 127.0.0.1 --port 8765 --no-open

pause
