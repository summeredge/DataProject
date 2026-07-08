from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
START_APP = ROOT / "start_app.bat"


def _start_app_text() -> str:
    return START_APP.read_text(encoding="utf-8")


def _non_echo_command_text() -> str:
    lines = []
    for line in _start_app_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("echo "):
            continue
        lines.append(stripped)
    return "\n".join(lines)


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


def test_start_app_required_dependency_check_uses_find_spec_not_imports():
    text = _start_app_text()
    assert 'import pandas, numpy' not in text
    assert 'import numpy, pandas' not in text
    assert re.search(r"find_spec\(['\"]pandas['\"]\)", text)
    assert re.search(r"find_spec\(['\"]numpy['\"]\)", text)


def test_start_app_does_not_auto_install_project_on_startup():
    text = _start_app_text()
    command_text = _non_echo_command_text()
    assert 'Installing required packages' not in text
    assert 'pip install -e .' not in command_text


def test_start_app_missing_required_dependencies_instructs_manual_install():
    text = _start_app_text()
    assert 'Required Python packages are missing' in text
    assert 'Please run' in text
    assert 'pip install -e .' in text


def test_start_app_still_uses_selected_python_command_for_checks():
    text = _start_app_text()
    assert '%PYTHON_CMD%' in text
    assert 'Checking Python dependencies' in text
    assert 'Starting local web app' in text
