from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Dict, Optional


class UploadMeta(BaseModel):
    device: str = Field(default="unknown")
    seq: Optional[int] = Field(default=None)
    ts: Optional[str] = Field(default=None)

    grade: str = Field(description="nm|sp|mp|hp|damaged")
    extra: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_form_value(cls, raw_meta: str) -> "UploadMeta":
        import json

        try:
            obj = json.loads(raw_meta) if raw_meta else {}
        except Exception as e:
            raise ValueError(f"Invalid JSON in meta: {e}") from e

        if not isinstance(obj, dict):
            raise ValueError("meta must be a JSON object")

        device = str(obj.pop("device", "unknown"))
        seq = obj.pop("seq", None)
        ts = obj.pop("ts", None)

        grade = obj.pop("grade", None)
        if grade is None:
            grade = obj.pop("condition", None)
        if grade is None:
            grade = obj.pop("label", None)
        if grade is None:
            raise ValueError("meta must include grade (or condition/label)")

        if seq is not None:
            try:
                seq = int(seq)
            except Exception:
                seq = None

        return cls(device=device, seq=seq, ts=ts, grade=str(grade), extra=obj)


class UploadResponse(BaseModel):
    ok: bool = True
    dataset_id: str
    grade: str
    front_path: str
    back_path: str
    sha256_front: str
    sha256_back: str
    indexed: bool = True
