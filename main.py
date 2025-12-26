# main.py
print("\n🚀 BACKEND STARTED – NANO BANANA MAINSTREAM 🚀")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import io
import os
import uuid
import base64
import imghdr
import traceback
from pathlib import Path
from typing import Optional, Any, Tuple, List

import requests
from PIL import Image

from fastapi import FastAPI, UploadFile, Form, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from google import genai
from google.genai import types

import r2_storage
from nano_banana_fallback import generate_with_nano_banana, NanoBananaError


# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"
UPLOAD_DIR = PUBLIC_DIR / "uploads"
THEMES_ROOT = PUBLIC_DIR / "themes"  # nano banana saves theme outputs here

PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
THEMES_ROOT.mkdir(parents=True, exist_ok=True)

print("📁 BASE_DIR:", BASE_DIR)
print("📁 PUBLIC_DIR:", PUBLIC_DIR)
print("📁 UPLOAD_DIR:", UPLOAD_DIR)
print("📁 THEMES_ROOT:", THEMES_ROOT)
print("☁️ R2_ENABLED:", bool(getattr(r2_storage, "R2_ENABLED", False)))


# --------------------------------------------------
# GEMINI CONFIG
# --------------------------------------------------
if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("❌ GEMINI_API_KEY is not set")

genai_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")

print("🧠 GEMINI_IMAGE_MODEL:", GEMINI_IMAGE_MODEL)


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

app.mount("/public", StaticFiles(directory=str(PUBLIC_DIR)), name="public")


# --------------------------------------------------
# UTILS
# --------------------------------------------------
def detect_mime(data: bytes) -> str:
    kind = imghdr.what(None, data)
    if kind == "png":
        return "image/png"
    if kind == "webp":
        return "image/webp"
    return "image/jpeg"


def validate_image_bytes(data: bytes) -> None:
    Image.open(io.BytesIO(data)).convert("RGB")


def normalize_to_jpeg_bytes(data: bytes, max_size: int = 1024) -> bytes:
    """
    Railway-safe:
    - Converts anything (webp/png/jpg) → JPEG bytes
    - Keeps size reasonable
    """
    img = Image.open(io.BytesIO(data)).convert("RGB")
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=92, optimize=True)
    return out.getvalue()


def load_image_bytes(path_or_url: str) -> bytes:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        r = requests.get(
            path_or_url,
            timeout=30,
            headers={"User-Agent": "gerox-backend/1.0"},
        )
        r.raise_for_status()
        return r.content
    return Path(path_or_url).read_bytes()


async def save_upload_async(file: UploadFile) -> Path:
    filename = file.filename or "upload.jpg"
    ext = (filename.split(".")[-1] or "jpg").lower()
    if ext not in ["jpg", "jpeg", "png", "webp"]:
        ext = "jpg"
    name = f"{uuid.uuid4().hex}.{ext}"
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
    request: Request,
    image_bytes: bytes,
    mime: str,
    extra: dict,
    r2_prefix: Optional[str] = None,
):
    """
    Used by /faceswap (nano banana).
    If R2 enabled -> imageUrl
    Else -> base64 in image
    """
    payload = {**extra, "success": True, "mime": mime}

    if bool(getattr(r2_storage, "R2_ENABLED", False)) and r2_prefix:
        url = _upload_to_r2(r2_prefix, image_bytes, content_type=mime)
        payload["imageUrl"] = url
        return JSONResponse(payload)

    payload["image"] = base64.b64encode(image_bytes).decode()
    return JSONResponse(payload)


def _maybe_b64_to_bytes(x: Any) -> Optional[bytes]:
    """
    Gemini SDK can return inline_data.data as:
      - bytes
      - base64 string
    """
    if x is None:
        return None
    if isinstance(x, (bytes, bytearray)):
        return bytes(x)
    if isinstance(x, str):
        try:
            return base64.b64decode(x)
        except Exception:
            return None
    return None


def _extract_image_and_text(obj: Any) -> Tuple[Optional[bytes], List[str]]:
    """
    Scans ALL candidates + ALL parts.
    Returns: (image_bytes_or_None, collected_texts)
    """
    texts: List[str] = []
    candidates = getattr(obj, "candidates", None) or []

    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) if content else None
        if not parts:
            continue

        for part in parts:
            t = getattr(part, "text", None)
            if t:
                texts.append(str(t))

            inline = getattr(part, "inline_data", None)
            if inline is not None:
                data = getattr(inline, "data", None)
                b = _maybe_b64_to_bytes(data)
                if b:
                    return b, texts

    return None, texts


# --------------------------------------------------
# ROOT + HEALTH
# --------------------------------------------------
@app.get("/")
def root():
    return {
        "ok": True,
        "service": "nano-banana-backend",
        "r2Enabled": bool(getattr(r2_storage, "R2_ENABLED", False)),
        "model": GEMINI_IMAGE_MODEL,
    }


@app.get("/health")
def health():
    return {"ok": True}


@app.get("//health")
def health_double_slash():
    return {"ok": True}


# --------------------------------------------------
# /faceswap ✅ Nano Banana MAINSTREAM
# --------------------------------------------------
@app.post("/faceswap")
async def faceswap(
    request: Request,
    source_img: UploadFile = File(...),
    theme_name: str | None = Form(None),
    # kept only for frontend compatibility
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

        raw_bytes = await source_img.read()
        validate_image_bytes(raw_bytes)

        # force jpeg before passing down
        face_jpeg = normalize_to_jpeg_bytes(raw_bytes, max_size=1024)

        theme = (theme_name or "default").strip()

        try:
            gen_bytes, gen_file = generate_with_nano_banana(
                face_bytes=face_jpeg,
                original_face_bytes=face_jpeg,
                theme_name=theme,
                themes_root=str(THEMES_ROOT),
            )
        except NanoBananaError as e:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": e.code, "message": e.message},
            )

        return _response_image(
            request=request,
            image_bytes=gen_bytes,
            mime="image/png",
            r2_prefix=f"outputs/nano_banana/{theme}",
            extra={"theme": theme, "used_target": gen_file, "engine": "nano_banana"},
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------
# /generate ✅ FIXED for Railway
# - Sends image as base64 string (like nano banana)
# - Parses inline_data robustly
# - Captures Gemini text when no image returned
# - Always returns imageUrl (R2 or /public/uploads)
# --------------------------------------------------
@app.post("/generate")
async def generate(
    request: Request,
    prompt: str = Form(...),
    face: UploadFile | None = File(None),
    faceUrl: str | None = Form(None),
):
    try:
        # -----------------------------
        # Load face bytes
        # -----------------------------
        if face:
            local_path = await save_upload_async(face)
            raw = local_path.read_bytes()
        elif faceUrl:
            raw = load_image_bytes(faceUrl)
        else:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "FACE_MISSING", "message": "Face image missing"},
            )

        validate_image_bytes(raw)

        # normalize to jpeg (stable for cv2 + uploads)
        jpeg_bytes = normalize_to_jpeg_bytes(raw, max_size=1024)

        # ✅ IMPORTANT: send base64 string (this is what works reliably on Railway)
        img_b64 = base64.b64encode(jpeg_bytes).decode("utf-8")

        print("🎨 Gemini image generation started")
        print("🧾 Prompt chars:", len(prompt or ""))

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part(text=(prompt or "")[:1500]),
                    types.Part(
                        inline_data={
                            "mime_type": "image/jpeg",
                            "data": img_b64,
                        }
                    ),
                ],
            )
        ]

        config = types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],  # allow text so we can debug blocks
            temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.4")),
            top_p=float(os.getenv("GEMINI_TOP_P", "0.8")),
            max_output_tokens=int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "2048")),
        )

        output_image = None
        debug_texts: List[str] = []

        # -----------------------------
        # 1) Non-stream attempt first (often more stable)
        # -----------------------------
        try:
            resp = genai_client.models.generate_content(
                model=GEMINI_IMAGE_MODEL,
                contents=contents,
                config=config,
            )
            output_image, debug_texts = _extract_image_and_text(resp)
        except Exception as e:
            print("⚠️ generate_content failed, will try stream:", e)

        # -----------------------------
        # 2) Stream fallback
        # -----------------------------
        if not output_image:
            for chunk in genai_client.models.generate_content_stream(
                model=GEMINI_IMAGE_MODEL,
                contents=contents,
                config=config,
            ):
                output_image, debug_texts = _extract_image_and_text(chunk)
                if output_image:
                    break

        if not output_image:
            msg = "No image returned by Gemini (no inline_data found)"
            if debug_texts:
                msg += " | Model text: " + " ".join(debug_texts)[:400]
            raise RuntimeError(msg)

        # -----------------------------
        # Return imageUrl (R2 preferred)
        # -----------------------------
        if bool(getattr(r2_storage, "R2_ENABLED", False)):
            url = _upload_to_r2("outputs/generate", output_image, "image/jpeg")
            print("✅ Gemini image uploaded to R2:", url)
            return {"success": True, "imageUrl": url, "model": GEMINI_IMAGE_MODEL}

        out_name = f"{uuid.uuid4().hex}.jpg"
        out_path = UPLOAD_DIR / out_name
        out_path.write_bytes(output_image)

        base_url = str(request.base_url).rstrip("/")
        image_url = f"{base_url}/public/uploads/{out_name}"
        print("✅ Gemini image generated:", image_url)

        return {"success": True, "imageUrl": image_url, "model": GEMINI_IMAGE_MODEL}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "GENERATION_FAILED", "message": str(e)},
        )
