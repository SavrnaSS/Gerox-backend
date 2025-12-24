import cv2
import numpy as np


def enhance_and_upscale(img, min_size=None):
    """
    CPU-safe image enhancement.
    - Accepts np.ndarray or bytes
    - Enhances sharpness + contrast
    - Upscales ONLY if image is smaller than min_size
    """

    # ---------- Decode if bytes ----------
    if isinstance(img, (bytes, bytearray)):
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)

    if not isinstance(img, np.ndarray):
        raise ValueError("enhance_and_upscale received invalid image")

    h, w = img.shape[:2]

    # ---------- Enhancement ----------
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)

    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    enhanced = cv2.filter2D(enhanced, -1, sharpen_kernel)

    # ---------- Smart upscale ----------
    if min_size:
        min_w, min_h = min_size
        if w < min_w or h < min_h:
            scale = max(min_w / w, min_h / h)
            enhanced = cv2.resize(
                enhanced,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_CUBIC
            )

    return enhanced
