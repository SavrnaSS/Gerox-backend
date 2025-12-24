import os
import traceback
from urllib.parse import urlparse, unquote

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from swap_engine import run_face_swap


app = FastAPI()

# 🔥 Your real frontend public path
FRONTEND_PUBLIC = "/Users/sumitsingh/Documents/Test/my-app/public"

# Allow all (local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def local_theme_path_from_url(url: str):
    """
    Convert a Next.js theme URL:
      http://localhost:3001/themes/5/clean_Generated%20Image%20Nov...
    → Local OS path:
      /Users/.../public/themes/5/clean_Generated Image Nov...
    """

    parsed = urlparse(url)
    path = parsed.path  # "/themes/5/clean_Generated%20Image..."

    if "/themes/" not in path:
        return None

    # Extract everything after /themes/
    relative = path.split("/themes/")[1]  # "5/clean_Generated%20Image..."
    # Decode %20 → space, %2C → comma, etc.
    relative = unquote(relative)

    local_path = os.path.join(FRONTEND_PUBLIC, "themes", relative)
    return local_path


@app.post("/faceswap")
async def faceswap(
    source_img: UploadFile = File(...),
    target_img: UploadFile = File(None),
    target_img_url: str = Form(None)
):
    print("\n----- BACKEND DEBUG INPUT -----")
    print("source_img:", source_img.filename)
    print("target_img:", target_img.filename if target_img else None)
    print("target_img_url:", target_img_url)
    print("--------------------------------\n")

    try:
        # -------------------------
        # READ SOURCE
        # -------------------------
        source_bytes = await source_img.read()
        if not source_bytes:
            raise Exception("Source image is empty.")

        # -------------------------
        # TARGET CASE 1 — FILE
        # -------------------------
        if target_img:
            print("📁 Using uploaded target_img file")
            target_bytes = await target_img.read()
            if not target_bytes:
                raise Exception("Uploaded target image is empty.")

        # -------------------------
        # TARGET CASE 2 — URL (theme)
        # -------------------------
        else:
            if not target_img_url:
                raise Exception("target_img_url missing.")

            # Convert URL → local filesystem path
            local_path = local_theme_path_from_url(target_img_url)
            print("LOCAL PATH =", local_path)

            if not local_path or not os.path.exists(local_path):
                raise Exception(f"Theme image NOT FOUND at: {local_path}")

            with open(local_path, "rb") as f:
                target_bytes = f.read()

            if not target_bytes:
                raise Exception("Local theme image is empty/corrupted.")

        # -------------------------
        # RUN FACE SWAP
        # -------------------------
        print("🧠 Running face swap engine...")
        result_bytes = run_face_swap(source_bytes, target_bytes)

        if not result_bytes:
            raise Exception("Swap engine returned no output.")

        print("✅ Face swap SUCCESS!")
        return Response(content=result_bytes, media_type="image/png")

    except Exception as e:
        print("🔥 BACKEND ERROR:", e)
        traceback.print_exc()
        raise HTTPException(500, f"Swap failed: {str(e)}")
