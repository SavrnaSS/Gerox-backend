# main.py (FULL updated file — keeps your working logic + fixes UI by returning R2 URLs when available)
print("\n🚨 BACKEND STARTED – EMBEDDING MATCH MODE 🚨")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import io
import os
import uuid
import base64
import imghdr
import traceback
from pathlib import Path
import threading

import requests
from PIL import Image

from fastapi import FastAPI, UploadFile, Form, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from google import genai

import swap_engine
import r2_storage
from r2_theme_store import sync_theme_to_local, normalize_theme_name

from theme_matcher import (
    load_theme_cache,
    pick_best_theme_image,
    clear_theme_cache,
)
from theme_cache_builder import (
    ensure_all_theme_caches,
    rebuild_single_theme_cache,
)
from nano_banana_fallback import (
    generate_with_nano_banana,
    NanoBananaError,
)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MIN_SIMILARITY = 0.20

BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"
THEMES_ROOT = PUBLIC_DIR / "themes"
UPLOAD_DIR = PUBLIC_DIR / "uploads"

THEMES_ROOT.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

print("📁 BASE_DIR:", BASE_DIR)
print("📁 PUBLIC_DIR:", PUBLIC_DIR)
print("📁 THEMES_ROOT:", THEMES_ROOT)
print("📁 UPLOAD_DIR:", UPLOAD_DIR)
print("☁️ R2_ENABLED:", getattr(r2_storage, "R2_ENABLED", False))

# --------------------------------------------------
# GEMINI CONFIG
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
    return "image/png" if kind == "png" else "image/jpeg"


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


def ensure_theme_ready(theme_name: str) -> Path:
    """
    Ensures public/themes/<theme>/ exists locally.
    If R2 enabled, sync from R2.
    Returns local theme dir path.
    """
    theme_key = normalize_theme_name(theme_name)
    local_dir = THEMES_ROOT / theme_key

    if getattr(r2_storage, "R2_ENABLED", False):
        # download from R2 (etag-skip)
        sync_theme_to_local(theme_key, THEMES_ROOT)

    local_dir.mkdir(parents=True, exist_ok=True)
    return local_dir


def _background_warmup():
    """
    Runs warmup without blocking startup.
    Helps avoid first-request timeouts on Railway.
    """
    try:
        print("🔥 Background warmup started...")

        # 1) Ensure inswapper exists (from R2 in your swap_engine)
        try:
            if hasattr(swap_engine, "ensure_inswapper_present"):
                swap_engine.ensure_inswapper_present()
        except Exception as e:
            print("⚠️ Warmup: inswapper preload failed:", e)

        # 2) Warm up InsightFace (downloads buffalo_l) once at boot (if your swap_engine has warmup)
        try:
            if hasattr(swap_engine, "warmup"):
                swap_engine.warmup()
        except Exception as e:
            print("⚠️ Warmup: swap_engine.warmup failed:", e)

        print("✅ Background warmup done.")
    except Exception as e:
        print("⚠️ Background warmup crashed:", e)


# --------------------------------------------------
# STARTUP (keep light!)
# --------------------------------------------------
@app.on_event("startup")
def _startup():
    # Optional theme cache build (can be heavy)
    if os.getenv("BUILD_THEME_CACHE_ON_STARTUP", "0") == "1":
        try:
            ensure_all_theme_caches()
        except Exception as e:
            print("⚠️ ensure_all_theme_caches failed:", e)

    # Non-blocking warmup to reduce first-request timeouts
    if os.getenv("BACKGROUND_WARMUP", "1") == "1":
        threading.Thread(target=_background_warmup, daemon=True).start()


# --------------------------------------------------
# ROOT + HEALTH (Railway-friendly)
# --------------------------------------------------
@app.get("/")
def root():
    return {"ok": True, "service": "faceswap-backend"}


@app.get("/health")
def health():
    return {"ok": True}


# some proxies hit //health
@app.get("//health")
def health_double_slash():
    return {"ok": True}


# --------------------------------------------------
# FACE SWAP
# --------------------------------------------------
@app.post("/faceswap")
async def faceswap(
    request: Request,
    source_img: UploadFile = File(...),
    theme_name: str | None = Form(None),
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

        # -----------------------------
        # READ + VALIDATE SOURCE
        # -----------------------------
        source_bytes = await source_img.read()
        validate_image_bytes(source_bytes)

        chosen_target_bytes: bytes | None = None
        chosen_target_name: str | None = None
        similarity: float | None = None
        theme_key: str | None = None

        # -----------------------------
        # 1) explicit target file
        # -----------------------------
        if target_img is not None:
            target_bytes = await target_img.read()
            validate_image_bytes(target_bytes)
            chosen_target_bytes = target_bytes
            chosen_target_name = target_img.filename or "uploaded_target"

        # -----------------------------
        # 2) explicit target url
        # -----------------------------
        elif target_img_url:
            target_bytes = load_image_bytes(target_img_url)
            validate_image_bytes(target_bytes)
            chosen_target_bytes = target_bytes
            chosen_target_name = target_img_url

        # -----------------------------
        # 3) theme flow
        # -----------------------------
        elif theme_name:
            theme_key = normalize_theme_name(theme_name)

            # Ensure theme files are present locally (from R2 if enabled)
            ensure_theme_ready(theme_key)

            # Get embedding
            _, face = swap_engine.extract_user_face(source_bytes)
            user_embedding = face.normed_embedding

            theme_faces = load_theme_cache(theme_key)
            best_file, similarity = pick_best_theme_image(user_embedding, theme_faces)

            print(f"🔍 Theme '{theme_key}' similarity score: {similarity}")

            # Nano Banana fallback (kept)
            if similarity is not None and similarity < MIN_SIMILARITY:
                print("⚠️ Low similarity → Nano Banana fallback")
                try:
                    gen_bytes, gen_file = generate_with_nano_banana(
                        face_bytes=source_bytes,
                        original_face_bytes=source_bytes,
                        theme_name=theme_key,
                        themes_root=str(THEMES_ROOT),
                    )

                    # Refresh cache (optional)
                    try:
                        rebuild_single_theme_cache(theme_key)
                        clear_theme_cache(theme_key)
                    except Exception:
                        pass

                    # ✅ If R2 enabled, return URL (small JSON) — prevents huge base64 + fixes UI
                    if getattr(r2_storage, "R2_ENABLED", False):
                        out_key = f"outputs/faceswap/{theme_key}/{uuid.uuid4().hex}.png"
                        url = r2_storage.put_bytes(out_key, gen_bytes, content_type="image/png")
                        return JSONResponse({
                            "success": True,
                            "theme": theme_key,
                            "used_target": gen_file,
                            "similarity": similarity,
                            "mime": "image/png",
                            "imageUrl": url,
                        })

                    # Fallback: base64 (kept)
                    return JSONResponse({
                        "success": True,
                        "theme": theme_key,
                        "used_target": gen_file,
                        "similarity": similarity,
                        "mime": "image/png",
                        "image": base64.b64encode(gen_bytes).decode(),
                    })

                except NanoBananaError:
                    return JSONResponse(
                        status_code=200,
                        content={
                            "success": False,
                            "similarity": similarity,
                            "message": "AI generation temporarily unavailable",
                        },
                    )

            target_path = THEMES_ROOT / theme_key / best_file
            if not target_path.exists():
                raise RuntimeError(f"Theme target not found locally: {target_path}")

            chosen_target_bytes = target_path.read_bytes()
            chosen_target_name = best_file

        else:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "TARGET_MISSING",
                    "message": "Provide target_img, target_img_url, or theme_name",
                },
            )

        # -----------------------------
        # RUN FACE SWAP
        # -----------------------------
        result_png = swap_engine.run_face_swap(source_bytes, chosen_target_bytes)

        # ✅ IMPORTANT: Upload to R2 and return URL (small JSON)
        if getattr(r2_storage, "R2_ENABLED", False):
            out_theme = theme_key or "custom"
            out_key = f"outputs/faceswap/{out_theme}/{uuid.uuid4().hex}.png"
            url = r2_storage.put_bytes(out_key, result_png, content_type="image/png")

            return JSONResponse({
                "success": True,
                "theme": theme_key or theme_name,
                "used_target": chosen_target_name,
                "similarity": similarity,
                "mime": "image/png",
                "imageUrl": url,
            })

        # fallback: still return base64 if R2 not enabled
        return JSONResponse({
            "success": True,
            "theme": theme_key or theme_name,
            "used_target": chosen_target_name,
            "similarity": similarity,
            "mime": detect_mime(result_png),
            "image": base64.b64encode(result_png).decode(),
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------
# GENERATE (GEMINI)
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
            return JSONResponse(status_code=400, content={"error": "Face image missing"})

        image_bytes = normalize_image_bytes(image_bytes)
        print("🎨 Gemini image generation started")

        contents = [
            {
                "role": "user",
                "parts": [
                    {"text": prompt[:1000]},
                    # keep your working logic as-is (your genai client accepts bytes here)
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

        out_name = f"{uuid.uuid4()}.jpg"

        # ✅ Best: upload to R2 and return URL
        if getattr(r2_storage, "R2_ENABLED", False):
            r2_key = f"outputs/generate/{out_name}"
            r2_url = r2_storage.put_bytes(r2_key, output_image, content_type="image/jpeg")
            print("✅ Gemini image uploaded to R2:", r2_url)
            return {"success": True, "imageUrl": r2_url, "model": "gemini-2.5-flash-image"}

        # Fallback: save locally and return correct Railway URL
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
