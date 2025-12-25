# main.py
print("\n🚀 BACKEND STARTED – NANO BANANA MAINSTREAM MODE 🚀")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import io
import os
import uuid
import base64
import imghdr
import traceback
from pathlib import Path
from typing import Optional

import requests
from PIL import Image

from fastapi import FastAPI, UploadFile, Form, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from google import genai

import r2_storage
from nano_banana_fallback import generate_with_nano_banana, NanoBananaError


# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"
UPLOAD_DIR = PUBLIC_DIR / "uploads"
THEMES_ROOT = PUBLIC_DIR / "themes"  # kept for compatibility (nano banana no longer depends on images)

PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
THEMES_ROOT.mkdir(parents=True, exist_ok=True)

# If 1, also return base64 even when R2 is enabled (useful for debugging).
ALSO_RETURN_BASE64 = os.getenv("ALSO_RETURN_BASE64", "0") == "1"

print("📁 BASE_DIR:", BASE_DIR)
print("📁 PUBLIC_DIR:", PUBLIC_DIR)
print("📁 UPLOAD_DIR:", UPLOAD_DIR)
print("📁 THEMES_ROOT:", THEMES_ROOT)
print("☁️ R2_ENABLED:", getattr(r2_storage, "R2_ENABLED", False))
print("☁️ ALSO_RETURN_BASE64:", ALSO_RETURN_BASE64)

# --------------------------------------------------
# GEMINI CONFIG (used for /generate)
# --------------------------------------------------
if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("❌ GEMINI_API_KEY is not set")

genai_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

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
def detect_mime(data: bytes) -> str:
    kind = imghdr.what(None, data)
    if kind == "png":
        return "image/png"
    if kind in ("jpg", "jpeg"):
        return "image/jpeg"
    return "application/octet-stream"


def validate_image_bytes(data: bytes) -> None:
    Image.open(io.BytesIO(data)).convert("RGB")


def normalize_image_bytes(data: bytes, max_size=768) -> bytes:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=92, optimize=True)
    return out.getvalue()


def load_image_bytes(path_or_url: str) -> bytes:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        r = requests.get(path_or_url, timeout=20)
        r.raise_for_status()
        return r.content
    return Path(path_or_url).read_bytes()


async def save_upload_async(file: UploadFile) -> Path:
    filename = file.filename or "upload.jpg"
    ext = filename.split(".")[-1].lower()
    if ext not in ["jpg", "jpeg", "png", "webp"]:
        ext = "jpg"
    name = f"{uuid.uuid4()}.{ext}"
    path = UPLOAD_DIR / name
    data = await file.read()
    path.write_bytes(data)
    return path


def _upload_to_r2(prefix: str, data: bytes, content_type: str) -> str:
    key = f"{prefix}/{uuid.uuid4().hex}"
    if content_type == "image/png" and not key.endswith(".png"):
        key += ".png"
    if content_type == "image/jpeg" and not (key.endswith(".jpg") or key.endswith(".jpeg")):
        key += ".jpg"
    return r2_storage.put_bytes(key, data, content_type=content_type)


def _response_image(
    *,
    image_bytes: bytes,
    mime: str,
    extra: dict,
    r2_prefix: Optional[str] = None,
):
    """
    - If R2 enabled -> returns imageUrl (+ optional base64)
    - Else -> returns base64 as 'image'
    """
    payload = {**extra, "success": True, "mime": mime}

    if getattr(r2_storage, "R2_ENABLED", False) and r2_prefix:
        url = _upload_to_r2(r2_prefix, image_bytes, content_type=mime)
        payload["imageUrl"] = url
        if ALSO_RETURN_BASE64:
            payload["image"] = base64.b64encode(image_bytes).decode()
        return JSONResponse(payload)

    payload["image"] = base64.b64encode(image_bytes).decode()
    return JSONResponse(payload)


# --------------------------------------------------
# ROOT + HEALTH
# --------------------------------------------------
@app.get("/")
def root():
    return {
        "ok": True,
        "service": "faceswap-backend",
        "mode": "nano-banana-mainstream",
        "r2Enabled": bool(getattr(r2_storage, "R2_ENABLED", False)),
    }


@app.get("/health")
def health():
    return {"ok": True}


# --------------------------------------------------
# FACE SWAP (NOW: Nano Banana mainstream for ALL themes)
# --------------------------------------------------
@app.post("/faceswap")
async def faceswap(
    request: Request,
    source_img: UploadFile = File(...),
    theme_name: str | None = Form(None),
    # kept for backward compatibility (ignored)
    target_img: UploadFile | None = File(None),
    target_img_url: str | None = Form(None),
):
    try:
        print("----- BACKEND DEBUG INPUT -----")
        print("source_img:", getattr(source_img, "filename", None))
        print("target_img:", getattr(target_img, "filename", None) if target_img else None)
        print("target_img_url:", target_img_url)
        print("theme_name:", theme_name)
        print("--------------------------------")

        source_bytes = await source_img.read()
        validate_image_bytes(source_bytes)

        theme_key = (theme_name or "default").strip()

        # ✅ MAINSTREAM: Always use Nano Banana generator
        gen_bytes, gen_file = generate_with_nano_banana(
            face_bytes=source_bytes,
            original_face_bytes=source_bytes,
            theme_name=theme_key,
            themes_root=str(THEMES_ROOT),
        )

        mime = detect_mime(gen_bytes)
        if mime not in ("image/png", "image/jpeg"):
            # force safe default
            mime = "image/png"

        return _response_image(
            image_bytes=gen_bytes,
            mime=mime,
            r2_prefix=f"outputs/faceswap/{theme_key}",
            extra={
                "theme": theme_key,
                "used_target": gen_file,
                "mode": "nano-banana",
            },
        )

    except NanoBananaError as e:
        return JSONResponse(
            status_code=200,
            content={"success": False, "error": "NANO_BANANA_UNAVAILABLE", "message": str(e)},
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------
# GENERATE (GEMINI) - keep as your working trending art pipeline
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
            local_path = await save_upload_async(face)
            image_bytes = local_path.read_bytes()
        elif faceUrl:
            image_bytes = load_image_bytes(faceUrl)
        else:
            return JSONResponse(status_code=400, content={"success": False, "error": "Face image missing"})

        image_bytes = normalize_image_bytes(image_bytes)
        print("🎨 Gemini image generation started")

        contents = [
            {
                "role": "user",
                "parts": [
                    {"text": prompt[:1000]},
                    {"inline_data": {"mime_type": "image/jpeg", "data": image_bytes}},
                ],
            }
        ]

        output_image = None
        for chunk in genai_client.models.generate_content_stream(
            model="gemini-2.5-flash-image",
            contents=contents,
        ):
            if not getattr(chunk, "candidates", None):
                continue
            for candidate in chunk.candidates:
                content = getattr(candidate, "content", None)
                parts = getattr(content, "parts", None) if content else None
                if not parts:
                    continue
                for part in parts:
                    inline = getattr(part, "inline_data", None)
                    if inline and getattr(inline, "data", None):
                        output_image = inline.data

        if not output_image:
            raise RuntimeError("No image returned by Gemini")

        # Prefer R2 URL
        if getattr(r2_storage, "R2_ENABLED", False):
            url = _upload_to_r2("outputs/generate", output_image, "image/jpeg")
            print("✅ Gemini image uploaded to R2:", url)
            return {"success": True, "imageUrl": url, "model": "gemini-2.5-flash-image"}

        # fallback: local file URL
        out_name = f"{uuid.uuid4()}.jpg"
        out_path = UPLOAD_DIR / out_name
        out_path.write_bytes(output_image)

        base_url = str(request.base_url).rstrip("/")
        image_url = f"{base_url}/public/uploads/{out_name}"
        print("✅ Gemini image generated:", image_url)

        return {"success": True, "imageUrl": image_url, "model": "gemini-2.5-flash-image"}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "GENERATION_FAILED", "message": str(e)},
        )
