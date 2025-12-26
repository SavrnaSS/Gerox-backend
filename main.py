# main.py
print("\n🚀 BACKEND STARTED – NANO BANANA ONLY MODE 🚀")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import io
import os
import uuid
import base64
import imghdr
import traceback
from pathlib import Path
from typing import Optional, Tuple

import requests
from PIL import Image

from fastapi import FastAPI, UploadFile, Form, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from google import genai
from google.genai import types

# ✅ Keep your existing nano banana logic module AS-IS (no changes here)
from nano_banana_fallback import generate_with_nano_banana, NanoBananaError

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"
UPLOAD_DIR = PUBLIC_DIR / "uploads"
THEMES_ROOT = PUBLIC_DIR / "themes"  # Nano banana module saves theme images here

PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
THEMES_ROOT.mkdir(parents=True, exist_ok=True)

print("📁 BASE_DIR:", BASE_DIR)
print("📁 PUBLIC_DIR:", PUBLIC_DIR)
print("📁 UPLOAD_DIR:", UPLOAD_DIR)
print("📁 THEMES_ROOT:", THEMES_ROOT)

# --------------------------------------------------
# GEMINI CONFIG
# --------------------------------------------------
if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("❌ GEMINI_API_KEY is not set")

genai_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Model id confirmed in Google's model list
# (Gemini 2.5 Flash Image supports image output)
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")

# Return base64 too (helps frontend if URL fails)
ALSO_RETURN_BASE64 = os.getenv("ALSO_RETURN_BASE64", "1") == "1"

# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve /public/... from absolute PUBLIC_DIR
app.mount("/public", StaticFiles(directory=str(PUBLIC_DIR)), name="public")

# --------------------------------------------------
# UTILS
# --------------------------------------------------
def _detect_mime(data: bytes) -> str:
    kind = imghdr.what(None, data)
    if kind == "png":
        return "image/png"
    if kind in ("jpg", "jpeg"):
        return "image/jpeg"
    # default
    return "image/jpeg"


def _validate_image_bytes(data: bytes) -> None:
    Image.open(io.BytesIO(data)).convert("RGB")


def _normalize_image_bytes(data: bytes, max_size: int = 1024) -> bytes:
    """
    Keeps uploads reasonably small for Gemini.
    """
    img = Image.open(io.BytesIO(data)).convert("RGB")
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=92, optimize=True)
    return out.getvalue()


def _load_image_bytes(path_or_url: str) -> bytes:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        r = requests.get(path_or_url, timeout=25)
        r.raise_for_status()
        return r.content
    return Path(path_or_url).read_bytes()


async def _save_upload_async(file: UploadFile) -> Path:
    filename = file.filename or "upload.jpg"
    ext = filename.split(".")[-1].lower()
    if ext not in ("jpg", "jpeg", "png", "webp"):
        ext = "jpg"
    name = f"{uuid.uuid4().hex}.{ext}"
    path = UPLOAD_DIR / name
    data = await file.read()
    path.write_bytes(data)
    return path


def _save_output_bytes_to_uploads(image_bytes: bytes, preferred_ext: Optional[str] = None) -> Tuple[str, Path]:
    mime = _detect_mime(image_bytes)
    if preferred_ext:
        ext = preferred_ext.lstrip(".")
    else:
        ext = "png" if mime == "image/png" else "jpg"

    out_name = f"{uuid.uuid4().hex}.{ext}"
    out_path = UPLOAD_DIR / out_name
    out_path.write_bytes(image_bytes)
    return out_name, out_path


def _public_url_for_upload(request: Request, filename: str) -> str:
    base_url = str(request.base_url).rstrip("/")
    return f"{base_url}/public/uploads/{filename}"


def _response_image(
    *,
    request: Request,
    image_bytes: bytes,
    extra: dict,
) -> JSONResponse:
    """
    Always returns:
      - imageUrl
      - url  (alias for frontend)
    Optionally also returns:
      - image (base64)
    """
    mime = _detect_mime(image_bytes)
    out_name, _ = _save_output_bytes_to_uploads(image_bytes)
    url = _public_url_for_upload(request, out_name)

    payload = {
        **extra,
        "success": True,
        "mime": mime,
        "imageUrl": url,
        "url": url,  # ✅ alias (fixes your hook expecting data.url)
        "model": GEMINI_IMAGE_MODEL,
    }

    if ALSO_RETURN_BASE64:
        payload["image"] = base64.b64encode(image_bytes).decode("utf-8")

    return JSONResponse(payload)


def _extract_first_image_from_stream(stream) -> bytes:
    """
    Robustly extract image bytes from Gemini streaming response.
    Handles bytes or base64 string in inline_data.data.
    """
    for chunk in stream:
        candidates = getattr(chunk, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if not parts:
                continue
            for part in parts:
                inline = getattr(part, "inline_data", None)
                if not inline:
                    continue
                data = getattr(inline, "data", None)
                if not data:
                    continue

                # data can be bytes OR base64 string depending on SDK / mode
                if isinstance(data, (bytes, bytearray)):
                    return bytes(data)

                if isinstance(data, str):
                    try:
                        return base64.b64decode(data)
                    except Exception:
                        # Sometimes it's already raw-ish; last resort:
                        return data.encode("utf-8")

    raise RuntimeError("No image returned from Gemini stream")


def _gemini_generate_image_with_face(prompt: str, face_bytes: bytes) -> bytes:
    """
    Gemini image generation using a text prompt + identity reference image.
    """
    face_bytes = _normalize_image_bytes(face_bytes)

    # ✅ Use base64 for maximum compatibility (same style as your nano banana module)
    face_b64 = base64.b64encode(face_bytes).decode("utf-8")

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part(text=prompt[:4000]),
                types.Part(
                    inline_data={
                        "mime_type": "image/jpeg",
                        "data": face_b64,
                    }
                ),
            ],
        )
    ]

    config = types.GenerateContentConfig(
        response_modalities=["IMAGE"],  # ✅ critical: force image output
        temperature=0.4,
        top_p=0.9,
        max_output_tokens=2048,
    )

    stream = genai_client.models.generate_content_stream(
        model=GEMINI_IMAGE_MODEL,
        contents=contents,
        config=config,
    )

    return _extract_first_image_from_stream(stream)


# --------------------------------------------------
# ROOT + HEALTH
# --------------------------------------------------
@app.get("/")
def root():
    return {
        "ok": True,
        "service": "nano-banana-backend",
        "mode": "nano-banana-only",
        "model": GEMINI_IMAGE_MODEL,
        "alsoReturnBase64": ALSO_RETURN_BASE64,
    }

@app.get("/health")
def health():
    return {"ok": True}

@app.get("//health")
def health_double_slash():
    return {"ok": True}


# --------------------------------------------------
# "FACE SWAP" (BUT NOW IT'S NANO BANANA ONLY)
# Keeps endpoint so frontend doesn't break.
# --------------------------------------------------
@app.post("/faceswap")
async def faceswap(
    request: Request,
    source_img: UploadFile = File(...),
    theme_name: str | None = Form(None),
    # kept for compatibility but ignored:
    target_img: UploadFile | None = File(None),
    target_img_url: str | None = Form(None),
):
    try:
        print("----- BACKEND DEBUG INPUT -----")
        print("source_img:", getattr(source_img, "filename", None))
        print("theme_name:", theme_name)
        print("target_img:", getattr(target_img, "filename", None) if target_img else None)
        print("target_img_url:", target_img_url)
        print("--------------------------------")

        if not theme_name:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "THEME_MISSING",
                    "message": "theme_name is required in nano-banana-only mode",
                },
            )

        source_bytes = await source_img.read()
        _validate_image_bytes(source_bytes)

        # ✅ MAINSTREAM: always Nano Banana (no swap, no theme files picking)
        gen_bytes, gen_file = generate_with_nano_banana(
            face_bytes=source_bytes,
            original_face_bytes=source_bytes,
            theme_name=theme_name,
            themes_root=str(THEMES_ROOT),
        )

        return _response_image(
            request=request,
            image_bytes=gen_bytes,
            extra={
                "theme": theme_name,
                "used_target": gen_file,  # generated filename from nano banana module
                "mode": "nano-banana-only",
            },
        )

    except NanoBananaError as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=200,
            content={
                "success": False,
                "error": e.code,
                "message": e.message,
            },
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------
# GENERATE (PROMPT + FACE) — FIXED TO ALWAYS RETURN IMAGE
# --------------------------------------------------
@app.post("/generate")
async def generate(
    request: Request,
    prompt: str = Form(...),
    face: UploadFile | None = File(None),
    faceUrl: str | None = Form(None),
):
    try:
        if face:
            local_path = await _save_upload_async(face)
            face_bytes = local_path.read_bytes()
        elif faceUrl:
            face_bytes = _load_image_bytes(faceUrl)
        else:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "FACE_MISSING", "message": "Provide face or faceUrl"},
            )

        _validate_image_bytes(face_bytes)

        print("🎨 Gemini /generate started (image modality forced)")
        out_bytes = _gemini_generate_image_with_face(prompt, face_bytes)

        return _response_image(
            request=request,
            image_bytes=out_bytes,
            extra={
                "prompt": prompt[:200],
                "mode": "generate",
            },
        )

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "GENERATION_FAILED",
                "message": str(e),
            },
        )
