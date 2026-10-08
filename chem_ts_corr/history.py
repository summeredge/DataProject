"""Read-only disk history without inferred analysis results."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from chem_ts_corr.data import EXCEL_SUFFIXES, TEXT_SUFFIXES


def read_metadata(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def disk_size(path: Path) -> int:
    total = 0
    if path.exists():
        for item in path.rglob("*"):
            try:
                if item.is_file() and not item.is_symlink():
                    total += item.stat().st_size
            except OSError:
                continue
    return total


def record_time(value: object, path: Path) -> tuple[str, str]:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed.astimezone(timezone.utc).isoformat(), "metadata"
        except ValueError:
            pass
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(), "filesystem"


def query_history(uploads_dir: Path, runs_dir: Path) -> dict:
    uploads = []
    for path in uploads_dir.iterdir() if uploads_dir.exists() else []:
        try:
            if not path.is_file() or path.is_symlink() or path.suffix.lower() not in TEXT_SUFFIXES | EXCEL_SUFFIXES:
                continue
            metadata = read_metadata(path.with_suffix(".json"))
            uploaded_at, source = record_time(metadata.get("uploaded_at"), path)
            name = metadata.get("original_filename")
            uploads.append({
                "file_id": path.stem,
                "original_filename": name if isinstance(name, str) and name else None,
                "uploaded_at": uploaded_at, "time_source": source,
                "file_size": path.stat().st_size, "run_ids": [], "analysis_count": 0,
            })
        except OSError:
            continue
    by_id = {item["file_id"]: item for item in uploads}
    analyses = []
    for path in runs_dir.iterdir() if runs_dir.exists() else []:
        try:
            if not path.is_dir() or path.is_symlink():
                continue
            config_path = path / "run_config.json"
            config = read_metadata(config_path)
            created_at, source = record_time(config.get("created_at"), config_path if config_path.exists() else path)
            file_id = config.get("file_id")
            file_id = file_id if isinstance(file_id, str) else None
            upload = by_id.get(file_id)
            target = config.get("target")
            analyses.append({
                "run_id": path.name, "file_id": file_id,
                "target": target if isinstance(target, str) else None,
                "original_filename": upload["original_filename"] if upload else None,
                "upload_exists": upload is not None,
                "created_at": created_at, "time_source": source,
                "result_size": disk_size(path),
            })
            if upload is not None:
                upload["run_ids"].append(path.name)
                upload["analysis_count"] += 1
        except OSError:
            continue
    uploads.sort(key=lambda item: (item["uploaded_at"], item["file_id"]), reverse=True)
    analyses.sort(key=lambda item: (item["created_at"], item["run_id"]), reverse=True)
    upload_bytes = disk_size(uploads_dir)
    run_bytes = disk_size(runs_dir)
    return {
        "storage": {"upload_count": len(uploads), "analysis_count": len(analyses),
                    "upload_size": upload_bytes, "analysis_size": run_bytes,
                    "total_size": upload_bytes + run_bytes},
        "uploads": uploads, "analyses": analyses,
    }
