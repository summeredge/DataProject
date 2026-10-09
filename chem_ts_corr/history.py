"""Disk history and explicit manual cleanup without inferred results."""
from __future__ import annotations

import json
import re
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path

from chem_ts_corr.data import EXCEL_SUFFIXES, TEXT_SUFFIXES


def read_metadata(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _is_link(path: Path) -> bool:
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024)
    )


def disk_size(path: Path) -> int:
    total = 0
    pending = [path]
    while pending:
        item = pending.pop()
        try:
            if _is_link(item):
                continue
            if item.is_dir():
                pending.extend(item.iterdir())
            elif item.is_file():
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
    for path in uploads_dir.iterdir() if uploads_dir.exists() and not _is_link(uploads_dir) else []:
        try:
            if _is_link(path) or not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES | EXCEL_SUFFIXES:
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
    for path in runs_dir.iterdir() if runs_dir.exists() and not _is_link(runs_dir) else []:
        try:
            if _is_link(path) or not path.is_dir():
                continue
            config_path = path / "run_config.json"
            config = read_metadata(config_path)
            created_at, source = record_time(config.get("created_at"), config_path if config_path.exists() else path)
            file_id = config.get("file_id")
            file_id = file_id if isinstance(file_id, str) else None
            if not file_id:
                candidate = Path(str(config.get("input_path") or "")).stem
                file_id = candidate if re.fullmatch(r"[0-9a-f]{32}", candidate) else None
            upload = by_id.get(file_id)
            original_filename = upload["original_filename"] if upload else None
            if original_filename is None and file_id and re.fullmatch(r"[0-9a-f]{32}", file_id):
                original_filename = read_metadata(uploads_dir / f"{file_id}.json").get("original_filename")
            target = config.get("target")
            analyses.append({
                "run_id": path.name, "file_id": file_id,
                "target": target if isinstance(target, str) else None,
                "original_filename": original_filename if isinstance(original_filename, str) else None,
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


def _safe_cleanup_path(path: Path, root: Path) -> None:
    """Reject links/reparse points, including links inside a run directory."""
    if _is_link(root):
        raise ValueError("历史根目录不能是链接")
    if path.resolve().parent != root.resolve():
        raise ValueError("清理路径越界")
    pending = [path]
    while pending:
        item = pending.pop()
        if _is_link(item):
            raise ValueError("包含符号链接或重解析点，跳过清理")
        if item.is_dir():
            pending.extend(item.iterdir())


def cleanup_history(uploads_dir: Path, runs_dir: Path, *, mode: str,
                    file_ids: list[str], run_ids: list[str], execute: bool = False,
                    active: list[dict] = ()) -> dict:
    if mode not in {"selected", "all"}:
        raise ValueError("不支持的清理模式")
    for value in file_ids + run_ids:
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{32}", value):
            raise ValueError("Invalid history id")
    history = query_history(uploads_dir, runs_dir)
    skipped = []
    if mode == "all":
        file_ids, run_ids = [], []
        for root, kind in [(uploads_dir, "upload"), (runs_dir, "analysis")]:
            for path in root.iterdir() if root.exists() else []:
                identifier = path.stem if kind == "upload" else path.name
                valid = re.fullmatch(r"[0-9a-f]{32}", identifier)
                valid = valid and (kind == "analysis" or path.suffix.lower() in TEXT_SUFFIXES | EXCEL_SUFFIXES | {".json"})
                if valid:
                    (file_ids if kind == "upload" else run_ids).append(identifier)
                else:
                    skipped.append({"kind": kind, "id": path.name, "reason": "非可管理历史文件，已跳过"})
    file_ids, run_ids = sorted(set(file_ids)), sorted(set(run_ids))
    targets = {}
    conflicts = list(skipped)
    if mode == "all" and active:
        conflicts.append({"kind": "operation", "id": "all", "reason": "存在正在执行的操作，不能清空全部"})
    sizes = {"upload": 0, "analysis": 0}
    for kind, identifiers, root in [("analysis", run_ids, runs_dir), ("upload", file_ids, uploads_dir)]:
        for identifier in identifiers:
            if kind == "analysis":
                paths = [root / identifier]
            else:
                paths = [p for p in root.iterdir() if p.stem == identifier and p.suffix.lower() in TEXT_SUFFIXES | EXCEL_SUFFIXES | {".json"}] if root.exists() else []
            try:
                if not paths:
                    raise FileNotFoundError("历史文件不存在")
                for path in paths:
                    _safe_cleanup_path(path, root)
                    if (kind == "analysis") != path.is_dir():
                        raise ValueError("历史对象类型不匹配")
                if active and (mode == "all" or any(
                    (not op.get("run_id") and not op.get("file_id")) or
                    (kind == "upload" and not re.fullmatch(r"[0-9a-f]{32}", str(op.get("file_id") or ""))) or
                    (op.get("run_id") == identifier if kind == "analysis" else op.get("file_id") == identifier)
                    for op in active
                )):
                    raise ValueError("存在正在执行的关联操作，无法清理")
                if kind == "upload":
                    retained = [row for row in history["analyses"] if ("analysis", row["run_id"]) not in targets and (row["file_id"] == identifier or not re.fullmatch(r"[0-9a-f]{32}", row["file_id"] or ""))]
                    if retained:
                        conflicts.append({"kind": kind, "id": identifier, "reason": "仍被保留分析引用，或存在无法确认关联的分析", "related_analysis_count": len(retained)})
                        continue
                targets[(kind, identifier)] = paths
                sizes[kind] += sum(disk_size(p) if p.is_dir() else p.stat().st_size for p in paths)
            except (OSError, ValueError) as exc:
                conflicts.append({"kind": kind, "id": identifier, "reason": str(exc)})
    result = {"mode": mode, "execute": execute,
              "storage": {"upload_count": len(file_ids), "analysis_count": len(run_ids),
                          "upload_size": sizes["upload"], "analysis_size": sizes["analysis"],
                          "total_size": sum(sizes.values())},
              "conflicts": conflicts, "allowed": bool(targets) and not (mode == "all" and active),
              "deleted_file_ids": [], "deleted_run_ids": [], "released_size": 0}
    if not execute:
        return result
    before = disk_size(uploads_dir) + disk_size(runs_dir)
    if result["allowed"]:
        for (kind, identifier), paths in targets.items():
            try:
                if kind == "upload":
                    retained = [row for row in query_history(uploads_dir, runs_dir)["analyses"] if row["file_id"] == identifier or not re.fullmatch(r"[0-9a-f]{32}", row["file_id"] or "")]
                    if retained:
                        raise ValueError(f"仍有 {len(retained)} 条分析保留，上传数据未删除")
                for path in paths:
                    _safe_cleanup_path(path, uploads_dir if kind == "upload" else runs_dir)
                for path in sorted(paths, key=lambda p: p.suffix.lower() == ".json"):
                    if kind == "analysis":
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                result["deleted_run_ids" if kind == "analysis" else "deleted_file_ids"].append(identifier)
            except (OSError, ValueError) as exc:
                conflicts.append({"kind": kind, "id": identifier, "reason": str(exc)})
                if kind == "upload" and not any(p.stem == identifier and p.suffix.lower() in TEXT_SUFFIXES | EXCEL_SUFFIXES for p in uploads_dir.iterdir()):
                    result["deleted_file_ids"].append(identifier)
    result["released_size"] = max(0, before - disk_size(uploads_dir) - disk_size(runs_dir))
    result["complete"] = not conflicts
    return result
