import os
import cv2
import pickle
from insightface.app import FaceAnalysis

THEMES_DIR = "public/themes"
CACHE_DIR = "theme_cache"

os.makedirs(CACHE_DIR, exist_ok=True)

# ==================================================
# INSIGHTFACE (LOAD ONCE)
# ==================================================
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))


# ==================================================
# THEME-AWARE FACE AREA THRESHOLDS
# ==================================================
THEME_FACE_AREA_THRESHOLD = {
    "photoshoot_with_bike": 9000,
    "mountian_photoshoot": 10000,
    "selfie_with_cristiano_ronaldo": 12000,
    "selfie_with_scarlett_johnson": 14000,
}

DEFAULT_FACE_AREA_THRESHOLD = 25000


def face_area(face):
    x1, y1, x2, y2 = map(int, face.bbox)
    return (x2 - x1) * (y2 - y1)


# ==================================================
# BUILD CACHE FOR ONE THEME (SAFE & SMART)
# ==================================================
def build_cache_for_theme(theme_name: str):
    theme_path = os.path.join(THEMES_DIR, theme_name)
    cache_path = os.path.join(CACHE_DIR, f"{theme_name}.pkl")

    if not os.path.isdir(theme_path):
        print(f"⚠️ Theme directory missing: {theme_name}")
        return

    theme_key = theme_name.lower()
    min_area = THEME_FACE_AREA_THRESHOLD.get(
        theme_key,
        DEFAULT_FACE_AREA_THRESHOLD
    )

    data = []

    for file in os.listdir(theme_path):
        if not file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue

        img_path = os.path.join(theme_path, file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ Skipping unreadable image: {file}")
            continue

        faces = app.get(img)
        if not faces:
            print(f"⚠️ Skipping image (no face detected): {file}")
            continue

        face = max(faces, key=face_area)
        area = face_area(face)

        # --------------------------------------------------
        # 🔍 FACE TOO SMALL → TRY ZOOM PASS (CACHE ONLY)
        # --------------------------------------------------
        if area < min_area:
            x1, y1, x2, y2 = map(int, face.bbox)
            pad = int(0.6 * max(x2 - x1, y2 - y1))

            h, w = img.shape[:2]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            crop = img[
                max(0, cy - pad):min(h, cy + pad),
                max(0, cx - pad):min(w, cx + pad),
            ]

            faces_zoom = app.get(crop)
            if faces_zoom:
                face = max(faces_zoom, key=face_area)
                area = face_area(face)

        if area < min_area:
            print(
                f"⚠️ Skipping image (face too small "
                f"{area} < {min_area}): {file}"
            )
            continue

        data.append({
            "file": file,
            "embedding": face.normed_embedding
        })

        print(f"✅ Cached face: {file} (area {area})")

    # --------------------------------------------------
    # 🚫 DO NOT WIPE CACHE IF NOTHING FOUND
    # --------------------------------------------------
    if not data:
        print(
            f"⚠️ No valid faces found for theme: {theme_name} "
            f"(cache unchanged)"
        )
        return

    with open(cache_path, "wb") as f:
        pickle.dump(data, f)

    print(
        f"✅ Rebuilt cache for theme '{theme_name}' "
        f"with {len(data)} faces"
    )


# ==================================================
# ENSURE ALL THEME CACHES EXIST
# ==================================================
def ensure_all_theme_caches():
    for theme_name in os.listdir(THEMES_DIR):
        theme_path = os.path.join(THEMES_DIR, theme_name)
        if not os.path.isdir(theme_path):
            continue

        cache_path = os.path.join(CACHE_DIR, f"{theme_name}.pkl")
        if not os.path.exists(cache_path):
            print(f"⚠️ Cache missing for '{theme_name}', building…")
            build_cache_for_theme(theme_name)


# ==================================================
# 🔥 REBUILD SINGLE THEME CACHE (NANO BANANA SAFE)
# ==================================================
def rebuild_single_theme_cache(theme_name: str):
    """
    Rebuild cache for ONLY ONE THEME.
    Used after Nano Banana generation.
    - Theme-aware face size thresholds
    - Face zoom rescue for wide shots
    - Never wipes existing cache
    """
    print(f"🧠 Rebuilding cache for single theme: {theme_name}")
    build_cache_for_theme(theme_name)
