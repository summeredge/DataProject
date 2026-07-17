"""Paths for bundled read-only files and writable application data."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


APP_NAME = "ChemTsCorr"


def resource_path(*parts: str) -> Path:
    """Return a packaged resource path, or the project root while developing."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base.joinpath(*parts)


def user_data_dir() -> Path:
    """Return the stable writable location for desktop-generated files."""
    if getattr(sys, "frozen", False):
        local_app_data = os.environ.get("LOCALAPPDATA")
        base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
        return base / APP_NAME
    return resource_path("reports")


def desktop_log_path() -> Path:
    """Keep launcher logs outside the bundle and installation directory."""
    if getattr(sys, "frozen", False):
        return user_data_dir() / "logs" / "desktop-launcher.log"
    return Path(tempfile.gettempdir()) / "chem-ts-corr" / "desktop-launcher.log"
