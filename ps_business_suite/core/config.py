"""Configuration helpers for PS Business Suite."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from storage_paths import resolve_app_base_dir, resolve_database_path


@dataclass(slots=True)
class Settings:
    base_dir: Path
    db_path: Path
    date_format: str = "%d-%m-%Y"
    currency_symbol: str = "₹"
    uploads_dir: Path | None = None

    @property
    def uploads(self) -> Path:
        path = self.uploads_dir or (self.base_dir / "uploads")
        path.mkdir(parents=True, exist_ok=True)
        return path


load_dotenv()
_BASE = resolve_app_base_dir()
_DB_PATH = resolve_database_path()

settings = Settings(
    base_dir=_BASE,
    db_path=_DB_PATH,
    date_format=os.getenv("DATE_DISPLAY_FORMAT", "%d-%m-%Y"),
    currency_symbol=os.getenv("APP_CURRENCY_SYMBOL", "₹"),
    uploads_dir=Path(os.getenv("UPLOADS_DIR", _BASE / "uploads")).expanduser(),
)
