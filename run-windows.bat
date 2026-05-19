@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found. Run:  python -m venv .venv  and then:  .venv\Scripts\pip install -e .
  pause
  exit /b 1
)
if "%~1"=="" (
  echo Usage: run-windows.bat ^<folder^>
  echo Drag a folder of JPGs onto this script, or pass a folder path as the first argument.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m photo_autocrop.cli process-and-review "%~1"
pause
