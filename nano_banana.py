import os
import uuid
import replicate
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pathlib import Path

# ==================================================
# CONFIG
# ==================================================

UPLOAD_DIR = Path("public/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Replicate reads this automatically
if not os.getenv("REPLICATE_API_TOKEN"):
    raise RuntimeError("❌ REPLICATE_API_TOKEN is not set")

# ==================================================
# FASTAPI APP (ISOLATED)
# ==================================================

app = FastAPI(
    title="Nano Banana Generator",
    docs_url=False,        # important when mounting
    redoc_url=False,
    openapi_url=None,
)

# ==================================================
# HELPERS
# ==================================================

def save_upload(file: UploadFile) -> str:
    ext = file.filename.split(".")[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    filepath = UPLOAD_DIR / filename

    with open(filepath, "wb") as f:
        f.write(file.file.read())

    return str(filepath)

# ==================================================
# ROUTE
# ==================================================

@app.post("/generate")
async def generate(
    prompt: str = Form(...),
    face: UploadFile | None = File(None),
    faceUrl: str | None = Form(None),
):
    """
    Generate image using Replicate Nano Banana.
    Accepts:
      - prompt (string)
      - face (UploadFile) OR faceUrl (string)
    Returns:
      { imageUrl: string }
    """
    try:
        image_inputs = []

        # ---------- UPLOADED FACE ----------
        if face:
            local_path = save_upload(face)
            image_inputs.append(local_path)

        # ---------- FACE URL ----------
        elif faceUrl:
            image_inputs.append(faceUrl)

        else:
            return JSONResponse(
                status_code=400,
                content={"error": "Face image missing"},
            )

        # ---------- REPLICATE CALL ----------
        output = replicate.run(
            "google/nano-banana",
            input={
                "prompt": prompt,
                "image_input": image_inputs,
            }
        )

        # Replicate returns a file-like object
        return {
            "imageUrl": output.url,
            "model": "nano-banana",
        }

    except Exception as e:
        print("🔥 Nano Banana error:", e)
        return JSONResponse(
            status_code=500,
            content={
                "error": "GENERATION_FAILED",
                "message": str(e),
            },
        )
