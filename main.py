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

import requests
from PIL import Image

from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from google import genai

import swap_engine
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

# If you have r2 theme sync in main.py already, keep it.
# (not redefining your existing r2 logic here)

MIN_SIMILARITY = 0.20

BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"
THEMES_ROOT = PUBLIC_DIR / "themes"

UPLOAD_DIR = PUBLIC_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

print("📁 BASE_DIR:", BASE_DIR)
print("📁 PUBLIC_DIR:", PUBLIC_DIR)
print("📁 THEMES_ROOT:", THEMES_ROOT)
print("📁 UPLOAD_DIR:", UPLOAD_DIR)

if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("❌ GEMINI_API_KEY is not set")

genai_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/public", StaticFiles(directory=str(PUBLIC_DIR)), name="public")

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

@app.on_event("startup")
def _startup():
    # 🚫 DO NOT build caches by default on Railway.
    if os.getenv("BUILD_THEME_CACHE_ON_STARTUP", "0") == "1":
        try:
            ensure_all_theme_caches()
        except Exception as e:
            print("⚠️ ensure_all_theme_caches failed:", e)

    # Optional: preload inswapper at startup (usually keep OFF)
    if os.getenv("PRELOAD_INSWAPPER_ON_STARTUP", "0") == "1":
        try:
            swap_engine.ensure_inswapper_present()
            print("✅ inswapper_128 ready")
        except Exception as e:
            print("⚠️ inswapper preload failed:", e)

@app.get("/health")
def health():
    return {"ok": True}

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

        source_bytes = await source_img.read()
        validate_image_bytes(source_bytes)

        chosen_target_bytes: bytes | None = None
        chosen_target_name: str | None = None
        similarity: float | None = None

        if target_img is not None:
            target_bytes = await target_img.read()
            validate_image_bytes(target_bytes)
            chosen_target_bytes = target_bytes
            chosen_target_name = target_img.filename or "uploaded_target"

        elif target_img_url:
            target_bytes = load_image_bytes(target_img_url)
            validate_image_bytes(target_bytes)
            chosen_target_bytes = target_bytes
            chosen_target_name = target_img_url

        elif theme_name:
            # Your existing theme flow remains the same:
            _, face = swap_engine.extract_user_face(source_bytes)
            user_embedding = face.normed_embedding

            theme_faces = load_theme_cache(theme_name)
            best_file, similarity = pick_best_theme_image(user_embedding, theme_faces)
            print(f"🔍 Theme '{theme_name}' similarity score: {similarity}")

            if similarity is not None and similarity < MIN_SIMILARITY:
                print("⚠️ Low similarity → Nano Banana fallback")
                try:
                    gen_bytes, gen_file = generate_with_nano_banana(
                        face_bytes=source_bytes,
                        original_face_bytes=source_bytes,
                        theme_name=theme_name,
                        themes_root=str(THEMES_ROOT),
                    )

                    try:
                        rebuild_single_theme_cache(theme_name)
                        clear_theme_cache(theme_name)
                    except Exception:
                        pass

                    return JSONResponse({
                        "success": True,
                        "theme": theme_name,
                        "used_target": gen_file,
                        "similarity": similarity,
                        "mime": detect_mime(gen_bytes),
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

            # If you now load themes from R2, your existing logic should already
            # download to local cache. This path is still valid for local themes dir.
            target_path = THEMES_ROOT / theme_name / best_file
            if not target_path.exists():
                raise RuntimeError(f"Theme target not found: {target_path}")

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

        result_png = swap_engine.run_face_swap(source_bytes, chosen_target_bytes)

        return JSONResponse({
            "success": True,
            "theme": theme_name,
            "used_target": chosen_target_name,
            "similarity": similarity,
            "mime": detect_mime(result_png),
            "image": base64.b64encode(result_png).decode(),
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate(
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
        out_path = UPLOAD_DIR / out_name
        out_path.write_bytes(output_image)

        image_url = f"http://localhost:8000/public/uploads/{out_name}"
        print("✅ Gemini image generated:", image_url)

        return {"imageUrl": image_url, "model": "gemini-2.5-flash-image"}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": "GENERATION_FAILED", "message": str(e)},
        )
