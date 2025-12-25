import os
import cv2
import numpy as np
from pathlib import Path

import boto3
from botocore.config import Config

from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo

# ==================================================
# PATHS / CONFIG
# ==================================================
DEFAULT_INSWAPPER_PATH = Path.home() / ".insightface" / "models" / "inswapper_128.onnx"
INSWAPPER_PATH = Path(os.getenv("INSWAPPER_PATH", str(DEFAULT_INSWAPPER_PATH)))

INSWAPPER_R2_KEY = os.getenv("INSWAPPER_R2_KEY", "models/inswapper_128.onnx")

# R2 creds (same ones you use for themes)
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_BUCKET = os.getenv("R2_BUCKET")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")

R2_ENABLED = all([R2_ACCOUNT_ID, R2_BUCKET, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY])

def _r2_client():
    endpoint = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )

def ensure_inswapper_present() -> None:
    """
    Ensures inswapper_128.onnx exists on disk.
    - If missing, downloads from R2 using INSWAPPER_R2_KEY.
    """
    if INSWAPPER_PATH.exists():
        return

    # create folder
    INSWAPPER_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not R2_ENABLED:
        raise FileNotFoundError(
            f"inswapper_128.onnx NOT FOUND at {INSWAPPER_PATH} and R2 is not configured.\n"
            f"Set R2_ACCOUNT_ID, R2_BUCKET, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY and upload the model to R2.\n"
            f"Expected key: {INSWAPPER_R2_KEY}"
        )

    print(f"⬇️ Downloading inswapper model from R2: s3://{R2_BUCKET}/{INSWAPPER_R2_KEY}")
    s3 = _r2_client()
    s3.download_file(R2_BUCKET, INSWAPPER_R2_KEY, str(INSWAPPER_PATH))
    print(f"✅ inswapper model saved to: {INSWAPPER_PATH}")


# ==================================================
# INSIGHTFACE (LAZY LOADING)
# ==================================================
_face_app: FaceAnalysis | None = None
_swapper = None

def _get_face_app() -> FaceAnalysis:
    global _face_app
    if _face_app is None:
        print("🧠 Loading InsightFace FaceAnalysis (buffalo_l)...")
        _face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _face_app.prepare(ctx_id=0, det_size=(640, 640))
    return _face_app

def _get_swapper():
    global _swapper
    if _swapper is None:
        ensure_inswapper_present()
        print(f"🧠 Loading INSwapper model from: {INSWAPPER_PATH}")
        _swapper = model_zoo.get_model(str(INSWAPPER_PATH), providers=["CPUExecutionProvider"])
    return _swapper


# ==================================================
# PUBLIC API (your main.py expects these)
# ==================================================
def extract_user_face(source_bytes: bytes):
    """
    Returns (decoded_image_bgr, best_face)
    best_face has .normed_embedding used by your theme matcher.
    """
    img = cv2.imdecode(np.frombuffer(source_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid source image bytes")

    app = _get_face_app()
    faces = app.get(img)
    if not faces:
        raise ValueError("No face detected in source image")

    # pick largest face
    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )
    return img, face


def run_face_swap(source_bytes: bytes, target_bytes: bytes) -> bytes:
    """
    Returns PNG bytes of swapped target (paste_back=True).
    """
    src = cv2.imdecode(np.frombuffer(source_bytes, np.uint8), cv2.IMREAD_COLOR)
    tgt = cv2.imdecode(np.frombuffer(target_bytes, np.uint8), cv2.IMREAD_COLOR)

    if src is None or tgt is None:
        raise ValueError("Invalid input images")

    app = _get_face_app()
    swapper = _get_swapper()

    faces_src = app.get(src)
    faces_tgt = app.get(tgt)

    if len(faces_src) == 0 or len(faces_tgt) == 0:
        raise ValueError("Face not detected in one of the images")

    face_src = max(
        faces_src,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )
    face_tgt = max(
        faces_tgt,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )

    swapped = swapper.get(tgt, face_tgt, face_src, paste_back=True)

    ok, png = cv2.imencode(".png", swapped)
    if not ok:
        raise RuntimeError("Failed to encode PNG")
    return png.tobytes()
