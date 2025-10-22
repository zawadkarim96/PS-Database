# PS Mini CRM

## Release Notes
- v7.4: Fixed syntax and navigation chain. Dashboard shows only upcoming warranties (3 & 60 days). Removed Products/Orders pages, kept tables for import integrity. Customers page supports Needs (product/unit). DD-MM-YYYY display, flexible importer + Excel serial dates. Login safe rerun for all Streamlit versions.
- v7.5: Fixed selectbox errors; "Add Customer" now supports Purchase/Warranty fields (date, product, model, serial, unit) and auto-creates warranty.
- v7.6: Admins can bulk-delete customers from the Customers page with a select-all option.
- v7.7: Importer allows blank column mappings and optional skipping of blank rows. Dashboard counts active warranties by expiry date. Customer summaries merge duplicate records and report how many were combined.
- v7.8: Customer summary groups unnamed customers under "(blank)" so their contact info is still accessible.

## Run Without Touching the Command Line
These launchers create a dedicated virtual environment, install dependencies from `requirements.txt`, and then open the Streamlit app inside the native desktop shell (no browser required). You only need Python 3.9+ installed.

### Any Platform (single command or double-click)
Run `python run_app.py` from the repository root, or double-click the file in your file explorer. The script prepares the environment and launches the app inside a pywebview dialog titled **PS Service Software**.

### Windows
1. Double-click `run_app.bat`.
2. Wait for the first run to finish creating the virtual environment and installing dependencies. A desktop window titled **PS Service Software** will open automatically.
3. Reuse the same shortcut any time you want to reopen the app—no extra steps required.

### macOS & Linux
1. (First time only) Make the script executable: `chmod +x run_app.sh`.
2. Double-click `run_app.sh` in your file manager **or** run `./run_app.sh` from a terminal.
3. The script prepares the environment and opens the app inside a desktop window—no external browser involved.

## Build a One-Click Desktop App for Your Team
If your staff does not have Python installed, you can package the project into a standalone application and distribute it like a regular program.

1. On a machine with Python 3.9+ installed, open a terminal in the repository root.
2. Run `python build_executable.py`. The script creates a temporary virtual environment, installs PyInstaller, and produces a bundle inside `dist/PS Service Software/`.
3. Share the contents of `dist/PS Service Software/` with your staff. Windows users can double-click `PS Service Software.exe`; macOS and Linux users can run the executable from Finder/File Explorer or the terminal.

### Where user data lives
When launched from the packaged app, databases, uploads, and the Excel import template are stored in a writable folder per operating system:

- **Windows:** `%APPDATA%\ps-mini-crm`
- **macOS:** `~/Library/Application Support/ps-mini-crm`
- **Linux:** `${XDG_DATA_HOME:-~/.local/share}/ps-mini-crm`

Staff can back up or migrate the application by copying that folder. Deleting it resets the app to a clean state the next time the executable is opened.

## Troubleshooting
- If Python is not installed or not on your `PATH`, install it from [python.org](https://www.python.org/downloads/) (Windows) or via your package manager (macOS/Linux).
- To reset everything, delete the `.venv` folder and rerun the launcher to recreate a clean environment.
