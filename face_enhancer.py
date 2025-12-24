import cv2
import numpy as np
from config import ENABLE_FACE_UPSCALE, FACE_UPSCALE_SCALE
from enhancer import enhance_face_crop


def enhance_face_region(image: np.ndarray, face_bbox):
    """
    Enhance ONLY face region using GFPGAN + RealESRGAN
    """

    if not ENABLE_FACE_UPSCALE:
        return image

    x1, y1, x2, y2 = map(int, face_bbox)

    # 🔒 Safety clamp
    h, w, _ = image.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    face_crop = image[y1:y2, x1:x2]

    if face_crop.size == 0:
        return image

    # ===============================
    # GFPGAN (face restoration)
    # ===============================
    _, _, restored_face = gfpgan.enhance(
        face_crop,
        has_aligned=False,
        only_center_face=True,
        paste_back=False
    )

    if restored_face is None:
        restored_face = face_crop

    # ===============================
    # RealESRGAN (upscale face)
    # ===============================
    if FACE_UPSCALE_SCALE > 1:
        restored_face, _ = upscaler.enhance(
            restored_face,
            outscale=FACE_UPSCALE_SCALE
        )

        # Resize back to original bbox size
        restored_face = cv2.resize(
            restored_face,
            (x2 - x1, y2 - y1),
            interpolation=cv2.INTER_CUBIC
        )

    # ===============================
    # Paste back
    # ===============================
    output = image.copy()
    output[y1:y2, x1:x2] = restored_face

    return output
