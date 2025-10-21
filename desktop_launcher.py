#!/usr/bin/env python3
"""Desktop-friendly launcher for the PS Mini CRM Streamlit application.

This module is designed to be used in two contexts:

* Direct execution with ``python desktop_launcher.py`` for a locally cloned
  repository.
* As the entry point when the project is packaged into a standalone
  executable via PyInstaller (see ``build_executable.py``).

In both scenarios the launcher ensures that user data such as the SQLite
database, uploaded documents, and import templates live in a writable
location outside of the read-only application bundle. On first launch the
template Excel file is copied into that storage directory so staff members
always have access to it.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from streamlit.web import bootstrap


APP_SCRIPT_NAME = "app.py"
IMPORT_TEMPLATE_NAME = "import_template.xlsx"


def resource_path(relative_name: str) -> Path:
    """Return the path to a bundled resource for both source and frozen runs."""

    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS, relative_name)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent / relative_name


def determine_storage_dir() -> Path:
    """Choose a writable directory for app data based on the host platform."""

    if sys.platform.startswith("win"):
        base_dir = Path(os.getenv("APPDATA", Path.home()))
    elif sys.platform == "darwin":
        base_dir = Path.home() / "Library" / "Application Support"
    else:
        base_dir = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local" / "share"))

    storage_dir = base_dir / "ps-mini-crm"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


def ensure_template_file(storage_dir: Path) -> None:
    """Copy the Excel import template into the storage directory if needed."""

    template_source = resource_path(IMPORT_TEMPLATE_NAME)
    template_target = storage_dir / IMPORT_TEMPLATE_NAME

    if template_source.exists() and not template_target.exists():
        shutil.copy2(template_source, template_target)


def main() -> None:
    storage_dir = determine_storage_dir()
    ensure_template_file(storage_dir)

    os.environ.setdefault("APP_STORAGE_DIR", str(storage_dir))

    app_script = resource_path(APP_SCRIPT_NAME)

    # ``bootstrap.run`` starts the Streamlit runtime and blocks until exit.
    flag_options = {"server.headless": False}
    bootstrap.run(str(app_script), "", [], flag_options)


if __name__ == "__main__":
    main()
