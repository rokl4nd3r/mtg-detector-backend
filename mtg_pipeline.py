import base64
import io
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image
import cv2
from openai import OpenAI

# ----------------------------
# Prompts
# ----------------------------
ID_PROMPT = """
Identifique a carta de Magic: The Gathering (MTG) nas imagens (frente e/ou verso).
Retorne APENAS um objeto JSON (sem markdown, sem texto extra) com este formato:

{
  "name": "string",
  "set": "string",
  "collector_number": "string",
  "language": "string",
  "finish": "string",
  "confidence": 0.0,
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
    "confidence": 0.0,
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

# ----------------------------
# Helpers
# ----------------------------
def pil_to_jpeg_bytes(img: Image.Image, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()

def img_to_data_url(img: Image.Image) -> str:
    b = pil_to_jpeg_bytes(img)
    return "data:image/jpeg;base64," + base64.b64encode(b).decode("utf-8")

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
            pass
    return {}

def _clean_unknown(s: str) -> str:
    s = (s or "").strip()
    if s.lower() in {"unknown", "unk", "?", "n/a", "na", "null"}:
        return ""
    return s

BASIC_PT_TO_EN = {
    "floresta": "Forest",
    "ilha": "Island",
    "montanha": "Mountain",
    "pântano": "Swamp",
    "pantano": "Swamp",
    "planície": "Plains",
    "planicie": "Plains",
}

def oracle_name_for_scryfall(name: str, lang: str) -> str:
    name = (name or "").strip()
    lang = (lang or "").strip().lower()
    if not name:
        return name
    if lang == "pt":
        key = name.strip().lower()
        if key in BASIC_PT_TO_EN:
            return BASIC_PT_TO_EN[key]
    return name

# ----------------------------
# Crop (OpenCV)
# ----------------------------
def auto_crop_card_cv2(
    img: Image.Image,
    max_dim: int = 900,
    pad_frac: float = 0.12,
    border_margin_pct: float = 2.0,
    target_aspect: float = 63.0 / 88.0,
) -> Tuple[Image.Image, Dict[str, Any]]:
    debug: Dict[str, Any] = {"status": "ok"}
    if img is None:
        debug["status"] = "no_image"
        return img, debug

    W0, H0 = img.size
    scale = min(1.0, float(max_dim) / float(max(W0, H0)))
    small = img.resize((int(W0 * scale), int(H0 * scale))) if scale < 1.0 else img
    W, H = small.size
    border_margin = int(round((border_margin_pct / 100.0) * min(W, H)))

    bgr = cv2.cvtColor(np.array(small), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    v = float(np.median(blur))
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(blur, lower, upper)

    k = max(7, int(round(min(W, H) * 0.015)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed = cv2.dilate(closed, None, iterations=1)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = float(W * H)
    candidates = []

    for c in contours:
        area = float(cv2.contourArea(c))
        if area < 0.03 * img_area:
            continue

        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), angle = rect
        if rw <= 1 or rh <= 1:
            continue

        aspect = min(rw, rh) / max(rw, rh)
        diff = abs(aspect - target_aspect)

        box = cv2.boxPoints(rect).astype(np.int32)
        x, y, bw, bh = cv2.boundingRect(box)
        bbox_area = float(bw * bh)
        bbox_area_ratio = bbox_area / img_area

        border_touch = (
            x <= border_margin or y <= border_margin or
            (x + bw) >= (W - border_margin) or (y + bh) >= (H - border_margin)
        )

        extent = area / max(1.0, bbox_area)
        closeness = max(0.0, 1.0 - (diff / 0.25))
        score = bbox_area_ratio * (0.65 + 0.35 * closeness) * (0.55 if border_touch else 1.0) * extent

        candidates.append({"score": float(score), "bbox_small": (int(x), int(y), int(x + bw), int(y + bh))})

    candidates.sort(key=lambda d: d["score"], reverse=True)
    if not candidates:
        debug["status"] = "fallback_original_no_contour"
        debug["bbox"] = (0, 0, W0, H0)
        debug["scale"] = scale
        return img, debug

    l, t, r, b = candidates[0]["bbox_small"]
    bw = r - l
    bh = b - t
    pad = int(round(pad_frac * max(bw, bh)))
    l2 = max(0, l - pad)
    t2 = max(0, t - pad)
    r2 = min(W, r + pad)
    b2 = min(H, b + pad)

    def unscale(vv: int) -> int:
        return int(round(vv / scale)) if scale < 1.0 else int(vv)

    L = max(0, unscale(l2))
    T = max(0, unscale(t2))
    R = min(W0, unscale(r2))
    B = min(H0, unscale(b2))

    if (R - L) < 80 or (B - T) < 80:
        debug["status"] = "fallback_original_tiny_bbox"
        debug["bbox"] = (0, 0, W0, H0)
        debug["scale"] = scale
        return img, debug

    cropped = img.crop((L, T, R, B))
    debug.update({"status": "ok", "scale": scale, "bbox": (L, T, R, B), "pad_frac": pad_frac})
    return cropped, debug

# ----------------------------
# Scryfall (sem preço)
# ----------------------------
def scryfall_lookup(name: str, set_code: str = "", collector_number: str = "", lang: str = "") -> Dict[str, Any]:
    headers = {"User-Agent": "mtg-scanner/1.1"}
    base = "https://api.scryfall.com"

    def _get(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        r = requests.get(url, params=params, headers=headers, timeout=20)
        if r.status_code != 200:
            return {"_error": f"{r.status_code}", "_text": r.text[:800]}
        return r.json()

    name = (name or "").strip()
    set_code = _clean_unknown(set_code).lower()
    collector_number = (collector_number or "").strip()
    lang = _clean_unknown(lang).lower()

    if not name:
        return {"_error": "empty_name"}

    if set_code and collector_number:
        url = f"{base}/cards/{set_code}/{collector_number}"
        data = _get(url)
        if data.get("object") == "card":
            return data

    oracle = oracle_name_for_scryfall(name, lang)

    q = f'!"{oracle}"'
    if set_code:
        q += f" set:{set_code}"
    if lang:
        q += f" lang:{lang}"

    url = f"{base}/cards/search"
    data = _get(url, params={"q": q, "unique": "prints"})
    if isinstance(data, dict):
        data["_debug_query"] = q
    return data

# ----------------------------
# OpenAI Vision
# ----------------------------
def openai_vision_json(
    client: OpenAI,
    model: str,
    prompt: str,
    images: List[Image.Image],
    temperature: float = 0.0,
    max_tokens: int = 700,
) -> Dict[str, Any]:
    content = [{"type": "text", "text": prompt}]
    for im in images:
        if im is None:
            continue
        content.append({"type": "image_url", "image_url": {"url": img_to_data_url(im)}})

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = (resp.choices[0].message.content or "").strip()
    return _safe_json_parse(text)

def postprocess_id_result(d: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(d, dict):
        return {}
    d = dict(d)
    d["set"] = _clean_unknown(d.get("set", ""))
    d["language"] = _clean_unknown(d.get("language", ""))
    d["finish"] = _clean_unknown(d.get("finish", ""))
    return d

def postprocess_condition(d: Dict[str, Any], has_back: bool) -> Dict[str, Any]:
    if not isinstance(d, dict):
        return {}
    c = d.get("condition")
    if not isinstance(c, dict):
        return d

    c = dict(c)
    grade = (c.get("grade") or "unknown").strip().lower()
    c["grade"] = grade if grade in {"nm","lp","mp","hp","damaged","unknown"} else "unknown"

    try:
        conf = float(c.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    if c["grade"] != "unknown" and conf <= 0.0:
        conf = 0.01
    c["confidence"] = max(0.0, min(1.0, conf))

    if not has_back:
        c["needs_review"] = True
        c["marked_back"] = False
        c["notes"] = (c.get("notes") or "").strip() or "Sem foto do verso: não dá pra garantir se está marcada no deck."

    d = dict(d)
    d["condition"] = c
    return d
