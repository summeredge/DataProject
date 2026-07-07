from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
START_APP = ROOT / "start_app.bat"


def _start_app_text() -> str:
    return START_APP.read_text(encoding="utf-8")


def test_start_app_does_not_import_heavy_model_packages_on_startup():
    text = _start_app_text()
    assert 'import sklearn, shap' not in text
    assert 'import shap' not in text
    assert 'import sklearn' not in text


def test_start_app_does_not_auto_install_optional_model_packages_on_startup():
    text = _start_app_text()
    assert 'pip install scikit-learn shap' not in text
    assert 'pip install sklearn shap' not in text
    assert 'Installing model explanation packages' not in text


def test_start_app_uses_lightweight_optional_dependency_check_or_clear_message():
    text = _start_app_text()
    assert 'importlib.util.find_spec' in text
    assert 'optional' in text.lower() or '可选' in text


def test_start_app_still_uses_selected_python_command_for_checks():
    text = _start_app_text()
    assert '%PYTHON_CMD%' in text
    assert 'Checking Python dependencies' in text
    assert 'Starting local web app' in text
