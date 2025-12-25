# main.py
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
from typing import Optional

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
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.20"))

# If 1, sync full theme folder from R2 at request time (can be heavy).
SYNC_FULL_THEME_FROM_R2 = os.getenv("SYNC_FULL_THEME_FROM_R2", "0") == "1"

# If 1, also return base64 even when R2 is enabled (useful for local dev / debugging).
ALSO_RETURN_BASE64 = os.getenv("ALSO_RETURN_BASE64", "0") == "1"

BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"
THEMES_ROOT = PUBLIC_DIR / "themes"
UPLOAD_DIR = PUBLIC_DIR / "uploads"

THEMES_ROOT.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ✅ Theme cache dir (matches your error path /app/theme_cache/...)
THEME_CACHE_DIR = BASE_DIR / "theme_cache"
THEME_CACHE_DIR.mkdir(parents=True, exist_ok=True)

print("📁 BASE_DIR:", BASE_DIR)
print("📁 PUBLIC_DIR:", PUBLIC_DIR)
print("📁 THEMES_ROOT:", THEMES_ROOT)
print("📁 UPLOAD_DIR:", UPLOAD_DIR)
print("📁 THEME_CACHE_DIR:", THEME_CACHE_DIR)
print("☁️ R2_ENABLED:", getattr(r2_storage, "R2_ENABLED", False))
print("☁️ SYNC_FULL_THEME_FROM_R2:", SYNC_FULL_THEME_FROM_R2)
print("☁️ ALSO_RETURN_BASE64:", ALSO_RETURN_BASE64)

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


def _ensure_theme_dir(theme_key: str) -> Path:
    d = THEMES_ROOT / theme_key
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_theme_ready(theme_name: str) -> Path:
    """
    Ensures public/themes/<theme>/ exists locally.
    If SYNC_FULL_THEME_FROM_R2=1 and R2 enabled, sync all files from R2.
    """
    theme_key = normalize_theme_name(theme_name)
    local_dir = _ensure_theme_dir(theme_key)

    if getattr(r2_storage, "R2_ENABLED", False) and SYNC_FULL_THEME_FROM_R2:
        sync_theme_to_local(theme_key, THEMES_ROOT)

    return local_dir


def ensure_theme_file_local(theme_key: str, filename: str) -> Path:
    """
    Ensure a specific theme file exists locally.
    If missing and R2 enabled -> download ONLY that file: themes/<theme>/<filename>
    """
    theme_dir = _ensure_theme_dir(theme_key)
    local_path = theme_dir / filename
    if local_path.exists():
        return local_path

    if getattr(r2_storage, "R2_ENABLED", False):
        r2_key = f"themes/{theme_key}/{filename}"
        try:
            data = r2_storage.get_bytes(r2_key)
            local_path.write_bytes(data)
            return local_path
        except Exception as e:
            if SYNC_FULL_THEME_FROM_R2:
                try:
                    sync_theme_to_local(theme_key, THEMES_ROOT)
                    if local_path.exists():
                        return local_path
                except Exception:
                    pass
            raise RuntimeError(f"Theme file missing and failed to fetch from R2: {r2_key} ({e})")

    raise RuntimeError(f"Theme target not found locally: {local_path} (R2 disabled)")


def _upload_to_r2(prefix: str, data: bytes, content_type: str) -> str:
    """
    Upload bytes to R2 and return URL. Requires R2_ENABLED.
    """
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
    Standard response for images:
    - If R2 enabled -> returns imageUrl (+ optional base64 if ALSO_RETURN_BASE64=1)
    - Else -> returns base64 as 'image'
    """
    payload = {**extra, "success": True, "mime": mime}

    if getattr(r2_storage, "R2_ENABLED", False) and r2_prefix:
        url = _upload_to_r2(r2_prefix, image_bytes, content_type=mime)
        payload["imageUrl"] = url
        if ALSO_RETURN_BASE64:
            payload["image"] = base64.b64encode(image_bytes).decode()
        return JSONResponse(payload)

    # local/base64 fallback
    payload["image"] = base64.b64encode(image_bytes).decode()
    return JSONResponse(payload)


def _background_warmup():
    try:
        print("🔥 Background warmup started...")
        try:
            if hasattr(swap_engine, "ensure_inswapper_present"):
                swap_engine.ensure_inswapper_present()
        except Exception as e:
            print("⚠️ Warmup: inswapper preload failed:", e)

        try:
            if hasattr(swap_engine, "warmup"):
                swap_engine.warmup()
        except Exception as e:
            print("⚠️ Warmup: swap_engine.warmup failed:", e)

        print("✅ Background warmup done.")
    except Exception as e:
        print("⚠️ Background warmup crashed:", e)


# --------------------------------------------------
# ✅ THEME CACHE AUTO-ENSURE (NEW, fixes Railway cache-missing)
# --------------------------------------------------
_theme_cache_locks: dict[str, threading.Lock] = {}

def _get_theme_lock(theme_key: str) -> threading.Lock:
    lk = _theme_cache_locks.get(theme_key)
    if lk is None:
        lk = threading.Lock()
        _theme_cache_locks[theme_key] = lk
    return lk

def _theme_cache_path(theme_key: str) -> Path:
    return THEME_CACHE_DIR / f"{theme_key}.pkl"

def ensure_theme_cache_ready(theme_key: str) -> Path:
    """
    Ensure /app/theme_cache/<theme>.pkl exists.
    Strategy:
      1) If exists locally -> ok
      2) If R2 enabled -> try download prebuilt cache (optional)
      3) Else build locally:
           - make sure theme images exist locally (sync theme from R2)
           - rebuild_single_theme_cache(theme_key)
           - clear_theme_cache(theme_key) so matcher reloads fresh
    """
    cache_path = _theme_cache_path(theme_key)
    if cache_path.exists():
        return cache_path

    lock = _get_theme_lock(theme_key)
    with lock:
        # double-check after acquiring lock (race safe)
        if cache_path.exists():
            return cache_path

        # 1) Try downloading cache from R2 if you uploaded caches there
        if getattr(r2_storage, "R2_ENABLED", False):
            candidate_keys = [
                f"theme_cache/{theme_key}.pkl",
                f"theme-cache/{theme_key}.pkl",
                f"cache/theme_cache/{theme_key}.pkl",
            ]
            for k in candidate_keys:
                try:
                    print(f"⬇️ Trying to download theme cache from R2: {k}")
                    data = r2_storage.get_bytes(k)
                    cache_path.write_bytes(data)
                    print(f"✅ Theme cache downloaded: {cache_path}")
                    clear_theme_cache(theme_key)
                    return cache_path
                except Exception as e:
                    print(f"⚠️ Cache not found at {k}: {e}")

        # 2) Build cache locally (needs theme files)
        try:
            if getattr(r2_storage, "R2_ENABLED", False):
                print(f"⬇️ Cache missing → syncing theme locally for rebuild: {theme_key}")
                sync_theme_to_local(theme_key, THEMES_ROOT)

            print(f"🧱 Rebuilding theme cache for: {theme_key}")
            rebuild_single_theme_cache(theme_key)
            clear_theme_cache(theme_key)

            if cache_path.exists():
                print(f"✅ Theme cache rebuilt: {cache_path}")
                return cache_path

            raise RuntimeError(f"Rebuild finished but cache still missing: {cache_path}")

        except Exception as e:
            raise RuntimeError(f"Theme cache missing for '{theme_key}' and rebuild failed: {e}")


# --------------------------------------------------
# STARTUP
# --------------------------------------------------
@app.on_event("startup")
def _startup():
    if os.getenv("BUILD_THEME_CACHE_ON_STARTUP", "0") == "1":
        try:
            ensure_all_theme_caches()
        except Exception as e:
            print("⚠️ ensure_all_theme_caches failed:", e)

    if os.getenv("BACKGROUND_WARMUP", "1") == "1":
        threading.Thread(target=_background_warmup, daemon=True).start()


# --------------------------------------------------
# ROOT + HEALTH
# --------------------------------------------------
@app.get("/")
def root():
    return {
        "ok": True,
        "service": "faceswap-backend",
        "r2Enabled": bool(getattr(r2_storage, "R2_ENABLED", False)),
    }

@app.get("/health")
def health():
    return {"ok": True}

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

        source_bytes = await source_img.read()
        validate_image_bytes(source_bytes)

        chosen_target_bytes: Optional[bytes] = None
        chosen_target_name: Optional[str] = None
        similarity: Optional[float] = None
        theme_key: Optional[str] = None

        # 1) explicit target file
        if target_img is not None:
            target_bytes = await target_img.read()
            validate_image_bytes(target_bytes)
            chosen_target_bytes = target_bytes
            chosen_target_name = target_img.filename or "uploaded_target"

        # 2) explicit target url
        elif target_img_url:
            target_bytes = load_image_bytes(target_img_url)
            validate_image_bytes(target_bytes)
            chosen_target_bytes = target_bytes
            chosen_target_name = target_img_url

        # 3) theme flow
        elif theme_name:
            theme_key = normalize_theme_name(theme_name)
            ensure_theme_ready(theme_key)

            # ✅ NEW: ensure cache exists (download/build) before loading
            ensure_theme_cache_ready(theme_key)

            # embedding
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

                    try:
                        rebuild_single_theme_cache(theme_key)
                        clear_theme_cache(theme_key)
                    except Exception:
                        pass

                    return _response_image(
                        request=request,
                        image_bytes=gen_bytes,
                        mime="image/png",
                        r2_prefix=f"outputs/faceswap/{theme_key}",
                        extra={
                            "theme": theme_key,
                            "used_target": gen_file,
                            "similarity": similarity,
                        },
                    )

                except NanoBananaError:
                    # keep behavior
                    return JSONResponse(
                        status_code=200,
                        content={
                            "success": False,
                            "similarity": similarity,
                            "message": "AI generation temporarily unavailable",
                        },
                    )

            # IMPORTANT: if best_file is not on disk (e.g. gen_*.png), fetch from R2 on demand
            target_path = ensure_theme_file_local(theme_key, best_file)
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

        # RUN FACE SWAP
        result_png = swap_engine.run_face_swap(source_bytes, chosen_target_bytes)

        # Return R2 URL if available (fixes UI)
        return _response_image(
            request=request,
            image_bytes=result_png,
            mime="image/png",
            r2_prefix=f"outputs/faceswap/{theme_key or 'custom'}",
            extra={
                "theme": theme_key or theme_name,
                "used_target": chosen_target_name,
                "similarity": similarity,
            },
        )

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
