@echo off
setlocal

REM Change to the directory where this script lives so relative paths work.
cd /d "%~dp0"

REM Figure out which Python interpreter to use.
set "PYTHON_BOOTSTRAP="

for %%I in (python.exe python3.exe) do (
    where %%I >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_BOOTSTRAP=%%~nI"
        goto :have_python
    )
)

where py.exe >nul 2>nul
if not errorlevel 1 (
    py -3 --version >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_BOOTSTRAP=py -3"
        goto :have_python
    )
)

echo Python 3.9 or newer must be installed before the launcher can run.
echo Download it from https://www.python.org/downloads/windows/ and rerun this shortcut.
goto :error

:have_python

REM Create a virtual environment if it doesn't already exist.
if not exist .venv (
    echo Creating Python virtual environment...
    %PYTHON_BOOTSTRAP% -m venv .venv
    if errorlevel 1 goto :error
)

REM Activate the environment and make sure dependencies are installed.
call .venv\Scripts\activate.bat
if errorlevel 1 goto :error

python -m pip install --disable-pip-version-check --upgrade pip >nul
if errorlevel 1 goto :error
python -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto :error

REM Launch the Streamlit app.
streamlit run app.py
if errorlevel 1 goto :error

goto :eof

:error
echo.
echo Failed to prepare or launch the app. Please ensure Python is installed.
exit /b 1

:eof
endlocal
