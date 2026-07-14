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
%PYTHON_CMD% -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('pandas') and importlib.util.find_spec('numpy') else 1)" >nul 2>nul
if errorlevel 1 (
  echo Required Python packages are missing.
  echo Please run: %PYTHON_CMD% -m pip install -e .
  pause
  exit /b 1
)

echo Checking optional model explanation dependencies...
%PYTHON_CMD% -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('sklearn') and importlib.util.find_spec('shap') else 1)" >nul 2>nul
if errorlevel 1 (
  echo Optional model explanation packages were not found.
  echo The app will still start; model explanation features may be unavailable until scikit-learn and SHAP are installed.
)

echo Starting local web app...
echo Open this URL if the browser does not open automatically:
echo %APP_URL%

start "" powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command "Start-Sleep -Seconds 2; Start-Process '%APP_URL%'"

%PYTHON_CMD% -m chem_ts_corr.cli serve --host 127.0.0.1 --port 8765 --no-open

pause
