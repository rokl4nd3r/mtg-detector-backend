from __future__ import annotations

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, status, Header
from fastapi.responses import JSONResponse
from zoneinfo import ZoneInfo
from datetime import datetime

from .config import Settings
from .auth import require_bearer_token
from .models import UploadMeta, UploadResponse
from .storage import Storage, normalize_grade


def iso_now(tz_name: str) -> str:
    tz = ZoneInfo(tz_name)
    return datetime.now(tz).isoformat()


def create_app() -> FastAPI:
    settings = Settings.from_env()

    storage = Storage(
        root=settings.dataset_root,
        staging_dir=settings.staging_dir,
        dataset_dir=settings.dataset_dir,
        index_jsonl=settings.index_jsonl,
        max_upload_bytes=settings.max_upload_bytes,
        timezone=settings.timezone,
    )
    storage.init_layout()

    app = FastAPI(title="MTG Dataset Collector Backend", version="1.0.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "ts": iso_now(settings.timezone)}

    @app.post("/dataset/upload", response_model=UploadResponse)
    def dataset_upload(
        front: UploadFile = File(...),
        back: UploadFile = File(...),
        meta: str = Form(...),
        authorization: str | None = Header(default=None),
    ):
        require_bearer_token(settings, authorization=authorization)

        try:
            m = UploadMeta.from_form_value(meta)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

        try:
            grade = normalize_grade(m.grade)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

        dataset_id, dev_safe, seq6 = storage.make_dataset_id(m.device, m.seq)

        try:
            saved_front, saved_back = storage.save_pair(front, back, grade, dataset_id)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

        line = {
            "ts_server": iso_now(settings.timezone),
            "ts_client": m.ts,
            "device": dev_safe,
            "seq": seq6,
            "grade": grade,
            "dataset_id": dataset_id,
            "front": storage.relpath_from_root(saved_front.final_path),
            "back": storage.relpath_from_root(saved_back.final_path),
            "sha256_front": saved_front.sha256_hex,
            "sha256_back": saved_back.sha256_hex,
            "bytes_front": saved_front.bytes_written,
            "bytes_back": saved_back.bytes_written,
            "meta_extra": m.extra,
        }

        try:
            storage.append_index_jsonl(line)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Index write failed: {e}")

        return UploadResponse(
            dataset_id=dataset_id,
            grade=grade,
            front_path=line["front"],
            back_path=line["back"],
            sha256_front=saved_front.sha256_hex,
            sha256_back=saved_back.sha256_hex,
            indexed=True,
        )

    @app.exception_handler(HTTPException)
    def http_exc_handler(_, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"ok": False, "error": exc.detail})

    return app


app = create_app()
