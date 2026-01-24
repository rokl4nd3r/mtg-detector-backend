from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    dataset_root: Path
    staging_dir: Path
    dataset_dir: Path
    index_jsonl: Path

    bearer_tokens: tuple[str, ...]
    max_upload_bytes: int  # total por request (front+back)

    timezone: str
    bind_host: str
    bind_port: int

    @staticmethod
    def from_env() -> "Settings":
        root = Path(os.getenv("MTG_DATASET_ROOT", "/srv/mtg-dataset")).resolve()

        tokens_raw = os.getenv("MTG_BEARER_TOKENS", "").strip()
        tokens = tuple(t.strip() for t in tokens_raw.split(",") if t.strip())

        max_mb = int(os.getenv("MTG_MAX_UPLOAD_MB", "25"))
        max_bytes = max_mb * 1024 * 1024

        tz = os.getenv("MTG_TIMEZONE", "America/Sao_Paulo")

        bind_host = os.getenv("MTG_BIND_HOST", "127.0.0.1").strip()
        bind_port = int(os.getenv("MTG_BIND_PORT", "8000"))

        staging = root / "staging"
        dataset = root / "dataset"
        index = root / "dataset.jsonl"

        return Settings(
            dataset_root=root,
            staging_dir=staging,
            dataset_dir=dataset,
            index_jsonl=index,
            bearer_tokens=tokens,
            max_upload_bytes=max_bytes,
            timezone=tz,
            bind_host=bind_host,
            bind_port=bind_port,
        )
