import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np

import boto3
from botocore.config import Config

from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo

# ==================================================
# MODEL PATHS
# ==================================================
DEFAULT_INSWAPPER_PATH = Path.home() / ".insightface" / "models" / "inswapper_128.onnx"
INSWAPPER_PATH = Path(os.getenv("INSWAPPER_PATH", str(DEFAULT_INSWAPPER_PATH)))

INSWAPPER_R2_KEY = os.getenv("INSWAPPER_R2_KEY", "models/inswapper_128.onnx")

# ==================================================
# R2 CONFIG
# ==================================================
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
    Downloads from R2 if missing.
    """
    if INSWAPPER_PATH.exists():
        return

    INSWAPPER_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not R2_ENABLED:
        raise FileNotFoundError(
            f"inswapper_128.onnx missing at {INSWAPPER_PATH} and R2 not configured.\n"
            f"Upload to R2 key: {INSWAPPER_R2_KEY} and set R2_* env vars."
        )

    print(f"⬇️ Downloading inswapper from R2: s3://{R2_BUCKET}/{INSWAPPER_R2_KEY}", flush=True)
    s3 = _r2_client()
    s3.download_file(R2_BUCKET, INSWAPPER_R2_KEY, str(INSWAPPER_PATH))
    print(f"✅ inswapper saved: {INSWAPPER_PATH}", flush=True)


# ==================================================
# LAZY SINGLETONS (THREAD SAFE)
# ==================================================
_face_app = None
_swapper = None
_init_lock = threading.Lock()


def _get_face_app() -> FaceAnalysis:
    global _face_app
    if _face_app is not None:
        return _face_app

    with _init_lock:
        if _face_app is not None:
            return _face_app
        print("🧠 Loading InsightFace FaceAnalysis (buffalo_l)...", flush=True)
        _face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _face_app.prepare(ctx_id=0, det_size=(640, 640))
        print("✅ FaceAnalysis ready", flush=True)
        return _face_app


def _get_swapper():
    global _swapper
    if _swapper is not None:
        return _swapper

    with _init_lock:
        if _swapper is not None:
            return _swapper
        ensure_inswapper_present()
        print(f"🧠 Loading INSwapper model from: {INSWAPPER_PATH}", flush=True)
        _swapper = model_zoo.get_model(str(INSWAPPER_PATH), providers=["CPUExecutionProvider"])
        print("✅ INSwapper ready", flush=True)
        return _swapper


def warmup() -> None:
    """
    Preloads models so first /faceswap request doesn't time out on Railway.
    Safe to call multiple times.
    """
    t0 = time.perf_counter()
    try:
        ensure_inswapper_present()
    except Exception as e:
        # keep behavior: raise for missing model if no R2 configured
        print("⚠️ warmup: ensure_inswapper_present failed:", e, flush=True)
        raise

    _get_face_app()
    _get_swapper()
    dt = time.perf_counter() - t0
    print(f"✅ swap_engine warmup complete in {dt:.2f}s", flush=True)


# ==================================================
# PUBLIC API (used by main.py)
# ==================================================
def extract_user_face(source_bytes: bytes):
    img = cv2.imdecode(np.frombuffer(source_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid source image bytes")

    app = _get_face_app()
    faces = app.get(img)
    if not faces:
        raise ValueError("No face detected in source image")

    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )
    return img, face


def run_face_swap(source_bytes: bytes, target_bytes: bytes) -> bytes:
    src = cv2.imdecode(np.frombuffer(source_bytes, np.uint8), cv2.IMREAD_COLOR)
    tgt = cv2.imdecode(np.frombuffer(target_bytes, np.uint8), cv2.IMREAD_COLOR)

    if src is None or tgt is None:
        raise ValueError("Invalid input images")

    app = _get_face_app()
    swapper = _get_swapper()

    faces_src = app.get(src)
    faces_tgt = app.get(tgt)

    if not faces_src or not faces_tgt:
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
