"""Shared helpers for resolving PS Service Software storage locations."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

APP_STORAGE_SUBDIR = "ps-business-suite"
DEFAULT_DB_FILENAME = "ps_business_suite.db"
LEGACY_DB_FILENAME = "ps_crm.db"
_FALLBACK_DIR = Path("/tmp") / APP_STORAGE_SUBDIR
_fallback_in_use = False


def get_storage_dir() -> Path:
    """Return the default writable directory for application data."""

    if sys.platform.startswith("win"):
        base_dir = Path(os.getenv("APPDATA", Path.home()))
    elif sys.platform == "darwin":
        base_dir = Path.home() / "Library" / "Application Support"
    else:
        base_dir = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local" / "share"))

    storage_dir = base_dir / APP_STORAGE_SUBDIR
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


def resolve_app_base_dir() -> Path:
    """Return the configured storage directory, creating it on first use."""

    global _fallback_in_use

    preferred = Path(os.getenv("APP_STORAGE_DIR", get_storage_dir())).expanduser()
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        return preferred
    except OSError as exc:
        _fallback_in_use = True
        _FALLBACK_DIR.mkdir(parents=True, exist_ok=True)
        print(
            f"Warning: unable to use storage directory {preferred!s} ({exc}). "
            f"Falling back to {_FALLBACK_DIR!s}."
        )
        return _FALLBACK_DIR


def _promote_legacy_database(target: Path, legacy: Path) -> None:
    """Move/copy the legacy CRM database so existing data stays visible."""

    if target.exists() or not legacy.exists():
        return

    try:
        legacy.rename(target)
    except OSError:
        shutil.copy2(legacy, target)


def storage_fallback_active() -> bool:
    """Return ``True`` when a non-writable storage path forced a /tmp fallback."""

    return _fallback_in_use


def resolve_database_path(
    *,
    preferred_name: str = DEFAULT_DB_FILENAME,
    legacy_name: str = LEGACY_DB_FILENAME,
) -> Path:
    """Return the SQLite database path, migrating legacy files when needed."""

    db_path_env = os.getenv("DB_PATH")
    if db_path_env:
        db_path = Path(db_path_env).expanduser()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return db_path

    base_dir = resolve_app_base_dir()
    target = base_dir / preferred_name
    legacy = base_dir / legacy_name
    _promote_legacy_database(target, legacy)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target
