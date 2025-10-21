#!/usr/bin/env bash
set -euo pipefail

# Move to the directory containing this script so relative paths resolve correctly.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VENV=".venv"

# Determine which Python binary to use for creating the virtual environment.
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    else
        echo "Python is required but was not found on PATH." >&2
        exit 1
    fi
fi

if [ ! -d "$VENV" ]; then
    echo "Creating Python virtual environment..."
    "$PYTHON_BIN" -m venv "$VENV"
fi

# shellcheck source=/dev/null
if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
elif [ -f "$VENV/Scripts/activate" ]; then
    # Handle Windows Subsystem for Linux or Git Bash where Scripts exists.
    source "$VENV/Scripts/activate"
else
    echo "Could not activate virtual environment." >&2
    exit 1
fi

python -m pip install --disable-pip-version-check --upgrade pip >/dev/null
python -m pip install --disable-pip-version-check -r requirements.txt

exec streamlit run app.py
