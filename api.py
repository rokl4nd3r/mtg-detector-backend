from dotenv import load_dotenv
load_dotenv()

import os
import io
import re
import json
import base64
import traceback
import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY não configurada (coloque no .env)")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))

app = FastAPI()

ID_PROMPT = """
Identifique a carta de Magic: The Gathering (MTG) nas imagens (frente e/ou verso).
Retorne APENAS um objeto JSON (sem markdown, sem texto extra) com este formato:

{
  "name": "string",
  "set": "string",                 // set code (ex: lea, mh2). Se não tiver certeza, use "" (string vazia). NUNCA use "unknown".
  "collector_number": "string",    // se não existir/visível, ""
  "language": "string",            // en, pt, es, de, fr, it, ja, ko, ru, zh, ou ""
  "finish": "string",              // nonfoil | foil | etched | "" (se não souber)
  "confidence": 0.0,               // 0..1
  "needs_review": false,
  "notes": "string",
  "candidates": [
    {"name":"string","set":"string","collector_number":"string","confidence":0.0}
  ]
}

Regras:
- Use a imagem da FRENTE como principal.
- Se houver dúvidas, preencha candidates com 2-5 opções.
"""

CONDITION_PROMPT = """
Você é um avaliador de condição física de cartas MTG baseado em fotos (frente e verso).
Retorne APENAS um objeto JSON (sem markdown, sem texto extra) no formato:

{
  "condition": {
    "grade": "nm|lp|mp|hp|damaged|unknown",
    "confidence": 0.0,         // 0.01..1.0 (nunca 0)
    "needs_review": false,
    "marked_back": false,
    "signals": ["edge_whitening","scratches","crease","bent_corner","stain","writing","scuffing","dirt","other"],
    "notes": "string"
  }
}

REGRA OBRIGATÓRIA:
- Se o VERSO tiver qualquer marca/avaria que possa deixar a carta identificável face-down no deck,
  então grade DEVE ser "damaged" e marked_back DEVE ser true.

Se o verso não existir ou estiver ruim demais para avaliar, needs_review=true.
"""

def _safe_json_parse(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        blob = m.group(0)
        try:
            return json.loads(blob)
        except Exception:
            return {}
    return {}

def _clean_unknown(s: str) -> str:
    s = (s or "").strip()
    if s.lower() in {"unknown", "unk", "?", "n/a", "na", "null"}:
        return ""
    return s

def pil_from_bytes(b: bytes) -> Optional[Image.Image]:
    if not b:
        return None
    try:
        return Image.open(io.BytesIO(b)).convert("RGB")
    except Exception:
        return None

def pil_to_jpeg_bytes(img: Image.Image, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()

def img_to_data_url(img: Image.Image) -> str:
    b = pil_to_jpeg_bytes(img)
    return "data:image/jpeg;base64," + base64.b64encode(b).decode("utf-8")

def openai_vision_json(client: OpenAI, prompt: str, images: List[Image.Image]) -> Dict[str, Any]:
    content = [{"type": "text", "text": prompt}]
    for im in images:
        if im is None:
            continue
        content.append({"type": "image_url", "image_url": {"url": img_to_data_url(im)}})

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": content}],
        temperature=TEMPERATURE,
        max_tokens=800,
    )
    text = (resp.choices[0].message.content or "").strip()
    return _safe_json_parse(text)

@app.middleware("http")
async def catch_all(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        tb = traceback.format_exc()
        log.error("Unhandled error path=%s err=%r\n%s", request.url.path, e, tb)
        return JSONResponse(status_code=500, content={"status": "error", "detail": repr(e), "traceback": tb})

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": OPENAI_MODEL,
        "temp": TEMPERATURE
    }

@app.post("/analyze")
async def analyze(
    front: UploadFile = File(...),
    back: UploadFile | None = File(None),
):
    try:
        front_bytes = await front.read()
        back_bytes = await back.read() if back is not None else b""

        if not front_bytes:
            raise HTTPException(status_code=400, detail="front is empty")

        front_img = pil_from_bytes(front_bytes)
        back_img = pil_from_bytes(back_bytes) if back_bytes else None

        if front_img is None:
            raise HTTPException(status_code=400, detail="front não abriu como imagem (JPEG/PNG inválido)")

        log.info(
            "REQ: front=%s ct=%s bytes=%d | back=%s ct=%s bytes=%d",
            front.filename, front.content_type, len(front_bytes),
            (back.filename if back else None), (back.content_type if back else None), len(back_bytes),
        )

        client = OpenAI(api_key=OPENAI_API_KEY)

        imgs = [front_img] + ([back_img] if back_img is not None else [])

        id_result = openai_vision_json(client, ID_PROMPT, imgs)
        if isinstance(id_result, dict):
            id_result = dict(id_result)
            id_result["set"] = _clean_unknown(id_result.get("set", ""))
            id_result["language"] = _clean_unknown(id_result.get("language", ""))
            id_result["finish"] = _clean_unknown(id_result.get("finish", ""))

        cond_result = openai_vision_json(client, CONDITION_PROMPT, imgs)

        out = {
            "status": "ok",
            "openai": {
                **(id_result or {}),
                **(cond_result or {}),
            },
            "debug": {
                "front_bytes": len(front_bytes),
                "back_bytes": len(back_bytes),
                "front_ct": front.content_type,
                "back_ct": (back.content_type if back else None),
                "model": OPENAI_MODEL,
            },
        }
        return out

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        log.error("CRASH in /analyze err=%r\n%s", e, tb)
        return JSONResponse(status_code=500, content={"status": "error", "detail": repr(e), "traceback": tb})
