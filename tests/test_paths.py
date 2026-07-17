import sys
from pathlib import Path

from chem_ts_corr import paths


def test_resource_path_uses_project_root_during_development(monkeypatch):
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)

    assert paths.resource_path("chem_ts_corr") == Path(__file__).resolve().parents[1] / "chem_ts_corr"


def test_resource_path_uses_pyinstaller_bundle(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)

    assert paths.resource_path("defaults", "settings.json") == tmp_path / "defaults" / "settings.json"


def test_frozen_user_data_and_log_paths_use_local_app_data(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    assert paths.user_data_dir() == tmp_path / "ChemTsCorr"
    assert paths.desktop_log_path() == tmp_path / "ChemTsCorr" / "logs" / "desktop-launcher.log"
