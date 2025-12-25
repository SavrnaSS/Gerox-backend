print("\n🚨 BACKEND STARTED – EMBEDDING MATCH MODE 🚨")

# --------------------------------------------------
# SILENCE WARNINGS
# --------------------------------------------------
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------
# STANDARD LIBS
# --------------------------------------------------
import io
import os
import uuid
import base64
import imghdr
import traceback
from pathlib import Path

import requests
from PIL import Image

# --------------------------------------------------
# FASTAPI
# --------------------------------------------------
from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# --------------------------------------------------
# GEMINI (new SDK)
# --------------------------------------------------
from google import genai

# --------------------------------------------------
# INTERNAL IMPORTS
# --------------------------------------------------
import swap_engine
import r2_storage
from r2_theme_store import sync_theme_to_local

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

# ✅ Local cache where themes are synced from R2:
THEMES_ROOT = BASE_DIR / ".theme_cache"
THEMES_ROOT.mkdir(parents=True, exist_ok=True)

# Local uploads folder (kept for compatibility; output will go to R2)
UPLOAD_DIR = PUBLIC_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

print("📁 BASE_DIR:", BASE_DIR)
print("📁 PUBLIC_DIR:", PUBLIC_DIR)
print("📁 THEMES_ROOT (local theme cache):", THEMES_ROOT)
print("📁 UPLOAD_DIR:", UPLOAD_DIR)
print("☁️ R2_ENABLED:", r2_storage.R2_ENABLED)

# --------------------------------------------------
# GEMINI CONFIG
# --------------------------------------------------
if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("❌ GEMINI_API_KEY is not set")

genai_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# --------------------------------------------------
# ENSURE THEME CACHES EXIST
# --------------------------------------------------
# This is safe even if THEMES_ROOT is empty at startup
ensure_all_theme_caches()

# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # dev ok; lock down in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# STATIC FILES
# --------------------------------------------------
# Keep serving /public if you still need it in dev
app.mount("/public", StaticFiles(directory=str(PUBLIC_DIR)), name="public")

# --------------------------------------------------
# UTILS
# --------------------------------------------------
def detect_mime(data: bytes) -> str:
    kind = imghdr.what(None, data)
    return "image/png" if kind == "png" else "image/jpeg"


def validate_image_bytes(data: bytes) -> None:
    Image.open(io.BytesIO(data)).convert("RGB")  # throws if invalid


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


def ensure_theme_ready(theme_name: str):
    """
    1) Sync theme images from R2 -> THEMES_ROOT/theme_name/
    2) Ensure theme cache exists (build if missing)
    3) Clear in-memory theme cache if necessary
    """
    theme_key = theme_name.strip().lower()

    if not r2_storage.R2_ENABLED:
        raise RuntimeError("R2 is not enabled, but you requested R2 theme loading.")

    # Sync from R2 into local theme cache folder
    sync_theme_to_local(theme_key, THEMES_ROOT)

    # If cache is missing, build it now
    cache_path = BASE_DIR / "theme_cache" / f"{theme_key}.pkl"
    if not cache_path.exists():
        print(f"⚠️ Cache missing for '{theme_key}', building…")
        rebuild_single_theme_cache(theme_key)
        clear_theme_cache(theme_key)


# --------------------------------------------------
# HEALTH
# --------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True}


# --------------------------------------------------
# API: FACE SWAP
# Supports:
#  - source_img (file) required
#  - theme_name (string) OR
#  - target_img (file) OR target_img_url (string)
# --------------------------------------------------
@app.post("/faceswap")
async def faceswap(
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
        # 3) theme flow (R2 -> local cache)
        # -----------------------------
        elif theme_name:
            theme_key = theme_name.strip().lower()

            # ✅ Ensure theme images exist locally by syncing from R2,
            # ✅ ensure cache exists
            ensure_theme_ready(theme_key)

            # Get embedding from swap_engine helper
            _, face = swap_engine.extract_user_face(source_bytes)
            user_embedding = face.normed_embedding

            theme_faces = load_theme_cache(theme_key)
            best_file, similarity = pick_best_theme_image(user_embedding, theme_faces)

            print(f"🔍 Theme '{theme_key}' similarity score: {similarity}")

            # Nano Banana fallback
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

                    # Upload generated result to R2 outputs (optional)
                    image_url = None
                    if r2_storage.R2_ENABLED:
                        out_key = r2_storage.new_key(f"outputs/nanobanana/{theme_key}", ".png")
                        image_url = r2_storage.put_bytes(out_key, gen_bytes, content_type="image/png")

                    return JSONResponse({
                        "success": True,
                        "theme": theme_key,
                        "used_target": gen_file,
                        "similarity": similarity,
                        "mime": detect_mime(gen_bytes),
                        "image": base64.b64encode(gen_bytes).decode(),  # keep existing behavior
                        "imageUrl": image_url,  # new (non-breaking)
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

            # Normal theme swap path (now from local cache synced from R2)
            target_path = THEMES_ROOT / theme_key / best_file
            if not target_path.exists():
                raise RuntimeError(f"Theme target not found after R2 sync: {target_path}")

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

        # Upload swap output to R2 (recommended)
        image_url = None
        if r2_storage.R2_ENABLED:
            out_key = r2_storage.new_key(f"outputs/faceswap/{(theme_name or 'custom').strip().lower()}", ".png")
            image_url = r2_storage.put_bytes(out_key, result_png, content_type="image/png")

        return JSONResponse({
            "success": True,
            "theme": theme_name,
            "used_target": chosen_target_name,
            "similarity": similarity,
            "mime": detect_mime(result_png),
            "image": base64.b64encode(result_png).decode(),  # keep existing frontend working
            "imageUrl": image_url,  # new (non-breaking)
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------
# API: GENERATE (GEMINI IMAGE)
# --------------------------------------------------
@app.post("/generate")
async def generate(
    prompt: str = Form(...),
    face: UploadFile | None = File(None),
    faceUrl: str | None = Form(None),
):
    try:
        # -----------------------------
        # LOAD FACE IMAGE
        # -----------------------------
        if face:
            # ✅ Avoid local save requirement: read bytes directly
            image_bytes = await face.read()
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
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_bytes,
                        }
                    },
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
                        output_image = inline.data  # keep latest

        if not output_image:
            raise RuntimeError("No image returned by Gemini")

        # -----------------------------
        # STORE OUTPUT IN R2 (instead of local)
        # -----------------------------
        if not r2_storage.R2_ENABLED:
            # fallback (dev) — keep old behavior if you didn't set R2 env
            out_name = f"{uuid.uuid4()}.jpg"
            out_path = UPLOAD_DIR / out_name
            out_path.write_bytes(output_image)
            image_url = f"http://localhost:8000/public/uploads/{out_name}"
            print("✅ Gemini image generated (local):", image_url)
        else:
            out_key = r2_storage.new_key("outputs/generate", ".jpg")
            image_url = r2_storage.put_bytes(out_key, output_image, content_type="image/jpeg")
            print("✅ Gemini image generated (R2):", image_url)

        return {
            "imageUrl": image_url,
            "model": "gemini-2.5-flash-image",
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "error": "GENERATION_FAILED",
                "message": str(e),
            },
        )
