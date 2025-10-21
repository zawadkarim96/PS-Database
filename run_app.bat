@echo off
setlocal

REM Change to the directory where this script lives so relative paths work.
cd /d "%~dp0"

REM Create a virtual environment if it doesn't already exist.
if not exist .venv (
    echo Creating Python virtual environment...
    python -m venv .venv
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
