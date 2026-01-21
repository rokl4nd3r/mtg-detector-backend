import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse

# 1) Garante multipart instalado (senão dá 500 antes de entrar no endpoint)
try:
    import multipart  # noqa: F401
except Exception as e:
    raise RuntimeError("Dependência faltando: instale com `python -m pip install python-multipart`") from e

# 2) Garante API key (se você realmente precisar dela no backend)
key = os.getenv("OPENAI_API_KEY", "").strip()
if not key:
    raise RuntimeError("OPENAI_API_KEY não configurada")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

app = FastAPI()

# 3) Middleware para capturar erros que acontecem antes do endpoint
@app.middleware("http")
async def catch_all_exceptions(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        log.exception("Unhandled error (before/inside endpoint). path=%s", request.url.path)
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(
    front: UploadFile = File(...),
    back: UploadFile = File(None),
):
    front_bytes = await front.read()
    back_bytes = await back.read() if back is not None else b""

    if not front_bytes:
        raise HTTPException(status_code=400, detail="front is empty")

    log.info(
        "analyze: front=%s ct=%s bytes=%d | back=%s ct=%s bytes=%d",
        front.filename, front.content_type, len(front_bytes),
        (back.filename if back else None), (back.content_type if back else None), len(back_bytes),
    )

    return {
        "status": "ok",
        "front_bytes": len(front_bytes),
        "back_bytes": len(back_bytes),
        "front_ct": front.content_type,
        "back_ct": (back.content_type if back else None),
    }
