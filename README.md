# PS Mini CRM

## Release Notes
- v7.4: Fixed syntax and navigation chain. Dashboard shows only upcoming warranties (3 & 60 days). Removed Products/Orders pages, kept tables for import integrity. Customers page supports Needs (product/unit). DD-MM-YYYY display, flexible importer + Excel serial dates. Login safe rerun for all Streamlit versions.
- v7.5: Fixed selectbox errors; "Add Customer" now supports Purchase/Warranty fields (date, product, model, serial, unit) and auto-creates warranty.
- v7.6: Admins can bulk-delete customers from the Customers page with a select-all option.
- v7.7: Importer allows blank column mappings and optional skipping of blank rows. Dashboard counts active warranties by expiry date. Customer summaries merge duplicate records and report how many were combined.
- v7.8: Customer summary groups unnamed customers under "(blank)" so their contact info is still accessible.

## Run Without Touching the Command Line
These launchers create a dedicated virtual environment, install dependencies from `requirements.txt`, and then open the Streamlit app in your browser. You only need Python 3.9+ installed.

### Windows
1. Double-click `run_app.bat`.
2. Wait for the first run to finish creating the virtual environment and installing dependencies. A browser tab will open automatically.
3. Reuse the same shortcut any time you want to reopen the app—no extra steps required.

### macOS & Linux
1. (First time only) Make the script executable: `chmod +x run_app.sh`.
2. Double-click `run_app.sh` in your file manager **or** run `./run_app.sh` from a terminal.
3. The script prepares the environment and opens the app in your default browser.

## Troubleshooting
- If Python is not installed or not on your `PATH`, install it from [python.org](https://www.python.org/downloads/) (Windows) or via your package manager (macOS/Linux).
- To reset everything, delete the `.venv` folder and rerun the launcher to recreate a clean environment.
