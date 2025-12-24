import cv2
import numpy as np
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))


def detect_gender_hair_beard(image_bytes: bytes):
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise Exception("Invalid image")

    faces = app.get(img)
    if not faces:
        raise Exception("No face detected")

    # Largest face only
    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )

    # ---------------- Gender (InsightFace safe) ----------------
    gender_raw = face.gender
    if isinstance(gender_raw, (int, np.integer)):
        male_prob = 1.0 if gender_raw == 1 else 0.0
        female_prob = 1.0 if gender_raw == 0 else 0.0
    else:
        male_prob = float(gender_raw[0])
        female_prob = float(gender_raw[1])

    # ---------------- Face regions ----------------
    x1, y1, x2, y2 = map(int, face.bbox)
    face_h = y2 - y1

    # Hair (upper head only)
    hair_top = max(0, y1 - int(face_h * 0.9))
    hair_bottom = y1 + int(face_h * 0.15)

    # Beard (lower face)
    beard_top = y1 + int(face_h * 0.55)
    beard_bottom = y2

    hair_region = img[hair_top:hair_bottom, x1:x2]
    beard_region = img[beard_top:beard_bottom, x1:x2]

    # ---------------- Hair length ----------------
    long_hair = False
    hair_ratio = 0

    if hair_region.size > 0:
        gray = cv2.cvtColor(hair_region, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
        hair_ratio = np.count_nonzero(mask) / mask.size
        long_hair = hair_ratio > 0.38

    # ---------------- Beard detection ----------------
    beard_detected = False
    beard_ratio = 0

    if beard_region.size > 0:
        gray = cv2.cvtColor(beard_region, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 85, 255, cv2.THRESH_BINARY_INV)
        beard_ratio = np.count_nonzero(mask) / mask.size
        beard_detected = beard_ratio > 0.22

    # ---------------- HAIR COLOR (HSV – FIXED) ----------------
    hair_color = "unknown"

    if hair_region.size > 0:
        hsv = cv2.cvtColor(hair_region, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Remove shadows & highlights
        valid = (v > 40) & (v < 220) & (s > 40)

        if np.count_nonzero(valid) > 150:
            avg_h = np.mean(h[valid])
            avg_s = np.mean(s[valid])
            avg_v = np.mean(v[valid])

            if avg_v < 80:
                hair_color = "black"
            elif avg_h < 25:
                hair_color = "brown"
            elif avg_h < 35 and avg_v > 140:
                hair_color = "blonde"
            elif avg_h < 15 and avg_s > 120:
                hair_color = "red"
            else:
                hair_color = "brown"

    # ---------------- FINAL GENDER DECISION ----------------
    if beard_detected and male_prob > 0.6:
        gender = "male"
    elif long_hair and female_prob > 0.4:
        gender = "female"
    elif male_prob > 0.75:
        gender = "male"
    elif female_prob > 0.75:
        gender = "female"
    else:
        gender = "male"

    return {
        "gender": gender,
        "gender_raw": int(gender_raw),
        "male_prob": round(male_prob, 3),
        "female_prob": round(female_prob, 3),
        "long_hair": long_hair,
        "hair_ratio": round(hair_ratio, 3),
        "beard_detected": beard_detected,
        "beard_ratio": round(beard_ratio, 3),
        "hair_color": hair_color,
    }
