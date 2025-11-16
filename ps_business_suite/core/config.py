"""Configuration helpers for PS Business Suite."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from storage_paths import get_storage_dir


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
_DEFAULT_BASE = Path(get_storage_dir())
_BASE = Path(os.getenv("APP_STORAGE_DIR", _DEFAULT_BASE)).expanduser()
_BASE.mkdir(parents=True, exist_ok=True)
_DB_PATH = Path(os.getenv("DB_PATH", _BASE / "ps_business_suite.db")).expanduser()
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

settings = Settings(
    base_dir=_BASE,
    db_path=_DB_PATH,
    date_format=os.getenv("DATE_DISPLAY_FORMAT", "%d-%m-%Y"),
    currency_symbol=os.getenv("APP_CURRENCY_SYMBOL", "₹"),
    uploads_dir=Path(os.getenv("UPLOADS_DIR", _BASE / "uploads")).expanduser(),
)
