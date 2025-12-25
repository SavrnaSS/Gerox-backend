import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo

os.makedirs(os.path.expanduser("~/.insightface/models"), exist_ok=True)

INSWAPPER_PATH = os.path.expanduser("~/.insightface/models/inswapper_128.onnx")

if not os.path.exists(INSWAPPER_PATH):
    raise FileNotFoundError(f"inswapper_128.onnx NOT FOUND at {INSWAPPER_PATH}")

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

swapper = model_zoo.get_model(INSWAPPER_PATH, providers=["CPUExecutionProvider"])


def _decode_cv2(img_bytes: bytes):
    """Decode bytes into OpenCV BGR image."""
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def extract_user_face(source_bytes: bytes):
    """
    Extract the best face from source image and return:
      (cropped_face_bytes_jpg, face_obj)

    face_obj includes:
      - .bbox
      - .normed_embedding  ✅ used by your theme matcher
    """
    src = _decode_cv2(source_bytes)
    if src is None:
        raise ValueError("Invalid source image")

    faces = app.get(src)
    if not faces:
        raise ValueError("No face detected in source image")

    # Choose the largest face (most reliable)
    def area(f):
        x1, y1, x2, y2 = f.bbox
        return float((x2 - x1) * (y2 - y1))

    face = max(faces, key=area)

    # Crop for optional debugging/usage
    x1, y1, x2, y2 = face.bbox.astype(int)
    h, w = src.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop = src[y1:y2, x1:x2]
    if crop.size == 0:
        cropped_bytes = b""
    else:
        ok, buf = cv2.imencode(".jpg", crop)
        cropped_bytes = buf.tobytes() if ok else b""

    return cropped_bytes, face


def run_face_swap(source_bytes, target_bytes):
    src = _decode_cv2(source_bytes)
    tgt = _decode_cv2(target_bytes)

    if src is None or tgt is None:
        raise ValueError("Invalid input images")

    faces_src = app.get(src)
    faces_tgt = app.get(tgt)

    if len(faces_src) == 0 or len(faces_tgt) == 0:
        raise ValueError("Face not detected in one of the images")

    face_src = faces_src[0]
    face_tgt = faces_tgt[0]

    swapped = swapper.get(tgt, face_tgt, face_src, paste_back=True)

    _, png = cv2.imencode(".png", swapped)
    return png.tobytes()
