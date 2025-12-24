import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo

os.makedirs(os.path.expanduser("~/.insightface/models"), exist_ok=True)

INSWAPPER_PATH = os.path.expanduser(
    "~/.insightface/models/inswapper_128.onnx"
)

if not os.path.exists(INSWAPPER_PATH):
    raise FileNotFoundError(f"inswapper_128.onnx NOT FOUND at {INSWAPPER_PATH}")

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

swapper = model_zoo.get_model(INSWAPPER_PATH, providers=["CPUExecutionProvider"])


def run_face_swap(source_bytes, target_bytes):
    src = cv2.imdecode(np.frombuffer(source_bytes, np.uint8), cv2.IMREAD_COLOR)
    tgt = cv2.imdecode(np.frombuffer(target_bytes, np.uint8), cv2.IMREAD_COLOR)

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
