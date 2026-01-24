from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import os
import re
import secrets
from typing import Tuple
from zoneinfo import ZoneInfo
from datetime import datetime

import fcntl


_GRADE_MAP = {
    "nm": "nm",
    "near mint": "nm",
    "nearmint": "nm",
    "sp": "sp",
    "slightly played": "sp",
    "slightplayed": "sp",
    "mp": "mp",
    "moderately played": "mp",
    "moderateplayed": "mp",
    "hp": "hp",
    "heavily played": "hp",
    "heavyplayed": "hp",
    "d": "damaged",
    "damaged": "damaged",
    "damage": "damaged",
}


def normalize_grade(raw: str) -> str:
    key = (raw or "").strip().lower()
    key = key.replace("_", " ").replace("-", " ")
    key = re.sub(r"\s+", " ", key).strip()
    out = _GRADE_MAP.get(key)
    if not out:
        raise ValueError(f"Invalid grade '{raw}'. Allowed: nm, sp, mp, hp, damaged")
    return out


_DEVICE_SAFE = re.compile(r"[^a-zA-Z0-9._-]+")


def sanitize_device(raw: str) -> str:
    s = (raw or "unknown").strip()
    s = _DEVICE_SAFE.sub("_", s)
    s = s.strip("._-")
    return s or "unknown"


def now_stamp_mmm(tz_name: str) -> str:
    tz = ZoneInfo(tz_name)
    dt = datetime.now(tz)
    return dt.strftime("%Y%m%d_%H%M%S_") + f"{dt.microsecond // 1000:03d}"


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


@dataclass
class SavedFile:
    final_path: Path
    sha256_hex: str
    bytes_written: int


@dataclass
class Storage:
    root: Path
    staging_dir: Path
    dataset_dir: Path
    index_jsonl: Path
    max_upload_bytes: int  # total por request (front+back)
    timezone: str

    def init_layout(self) -> None:
        ensure_dirs(
            self.staging_dir,
            self.dataset_dir / "nm",
            self.dataset_dir / "sp",
            self.dataset_dir / "mp",
            self.dataset_dir / "hp",
            self.dataset_dir / "damaged",
        )
        if not self.index_jsonl.exists():
            self.index_jsonl.parent.mkdir(parents=True, exist_ok=True)
            self.index_jsonl.touch()

    def make_dataset_id(self, device: str, seq: int | None) -> Tuple[str, str, str]:
        stamp = now_stamp_mmm(self.timezone)
        dev = sanitize_device(device)
        if seq is None or seq < 0 or seq > 999999:
            seq6 = f"{secrets.randbelow(1_000_000):06d}"
        else:
            seq6 = f"{seq:06d}"
        dataset_id = f"{stamp}_{dev}_{seq6}"
        return dataset_id, dev, seq6

    def _write_stream_to_file(self, upload_file, tmp_path: Path, max_bytes: int) -> SavedFile:
        if max_bytes <= 0:
            raise ValueError("Upload exceeds limit (no remaining budget)")

        hasher = hashlib.sha256()
        written = 0
        chunk_size = 1024 * 1024

        tmp_path.parent.mkdir(parents=True, exist_ok=True)

        with open(tmp_path, "wb") as f:
            while True:
                chunk = upload_file.file.read(chunk_size)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise ValueError(f"Upload exceeds limit ({max_bytes} bytes remaining)")
                hasher.update(chunk)
                f.write(chunk)
            f.flush()
            os.fsync(f.fileno())

        return SavedFile(final_path=tmp_path, sha256_hex=hasher.hexdigest(), bytes_written=written)

    def save_pair(self, front_upload, back_upload, grade: str, dataset_id: str) -> Tuple[SavedFile, SavedFile]:
        grade = normalize_grade(grade)

        allowed_types = {"image/jpeg", "image/jpg", "image/png", "application/octet-stream"}
        if (front_upload.content_type or "").lower() not in allowed_types:
            raise ValueError(f"Unsupported front content_type: {front_upload.content_type}")
        if (back_upload.content_type or "").lower() not in allowed_types:
            raise ValueError(f"Unsupported back content_type: {back_upload.content_type}")

        front_ext = ".png" if (front_upload.content_type or "").lower() == "image/png" else ".jpg"
        back_ext = ".png" if (back_upload.content_type or "").lower() == "image/png" else ".jpg"

        stage_front = self.staging_dir / f"{dataset_id}_front{front_ext}.part"
        stage_back = self.staging_dir / f"{dataset_id}_back{back_ext}.part"

        final_dir = self.dataset_dir / grade
        final_front = final_dir / f"{dataset_id}_front{front_ext}"
        final_back = final_dir / f"{dataset_id}_back{back_ext}"

        def safe_unlink(p: Path) -> None:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

        try:
            sf_front = self._write_stream_to_file(front_upload, stage_front, self.max_upload_bytes)
            remaining = self.max_upload_bytes - sf_front.bytes_written
            sf_back = self._write_stream_to_file(back_upload, stage_back, remaining)
        except Exception:
            safe_unlink(stage_front)
            safe_unlink(stage_back)
            raise

        try:
            ensure_dirs(final_dir)
            os.replace(stage_front, final_front)
            os.replace(stage_back, final_back)
        except Exception:
            safe_unlink(stage_front)
            safe_unlink(stage_back)
            safe_unlink(final_front)
            safe_unlink(final_back)
            raise

        return (
            SavedFile(final_path=final_front, sha256_hex=sf_front.sha256_hex, bytes_written=sf_front.bytes_written),
            SavedFile(final_path=final_back, sha256_hex=sf_back.sha256_hex, bytes_written=sf_back.bytes_written),
        )

    def append_index_jsonl(self, line_obj: dict) -> None:
        import json

        self.index_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_jsonl, "a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(line_obj, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def relpath_from_root(self, p: Path) -> str:
        try:
            return str(p.relative_to(self.root))
        except Exception:
            return str(p)
