import os
import cv2
import pickle
from pathlib import Path
from insightface.app import FaceAnalysis

# --------------------------------------------------
# PATHS (robust)
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
THEMES_DIR = Path(os.getenv("THEMES_DIR", BASE_DIR / "public" / "themes"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", BASE_DIR / "theme_cache"))

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
# INSIGHTFACE (LAZY LOAD)
# ==================================================
_face_app = None

def get_face_app():
    global _face_app
    if _face_app is None:
        print("🧠 Loading InsightFace for cache builder (buffalo_l)...")
        _face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        # ✅ faster + enough for cache building
        _face_app.prepare(ctx_id=0, det_size=(320, 320))
    return _face_app

# ==================================================
# THEME-AWARE FACE AREA THRESHOLDS
# (keep your special cases, but lower default)
# ==================================================
THEME_FACE_AREA_THRESHOLD = {
    "photoshoot_with_bike": 9000,
    "mountian_photoshoot": 10000,
    "selfie_with_cristiano_ronaldo": 12000,
    "selfie_with_scarlett_johnson": 14000,
}

# ✅ IMPORTANT: old default 25000 was too strict
DEFAULT_FACE_AREA_THRESHOLD = int(os.getenv("DEFAULT_FACE_AREA_THRESHOLD", "9000"))

def face_area(face):
    x1, y1, x2, y2 = map(int, face.bbox)
    return (x2 - x1) * (y2 - y1)

# ==================================================
# BUILD CACHE FOR ONE THEME
# ==================================================
def build_cache_for_theme(theme_name: str):
    theme_name = (theme_name or "").strip()
    theme_path = THEMES_DIR / theme_name
    cache_path = CACHE_DIR / f"{theme_name}.pkl"

    if not theme_path.is_dir():
        print(f"⚠️ Theme directory missing: {theme_name} ({theme_path})")
        # ✅ still write empty cache so backend won't loop forever
        with open(cache_path, "wb") as f:
            pickle.dump([], f)
        print(f"✅ Wrote EMPTY cache (theme missing) -> {cache_path}")
        return

    theme_key = theme_name.lower()
    min_area = THEME_FACE_AREA_THRESHOLD.get(theme_key, DEFAULT_FACE_AREA_THRESHOLD)

    data = []
    app = get_face_app()

    files = sorted([p for p in theme_path.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]])
    if not files:
        print(f"⚠️ No images in theme folder: {theme_name}")
        with open(cache_path, "wb") as f:
            pickle.dump([], f)
        print(f"✅ Wrote EMPTY cache (no images) -> {cache_path}")
        return

    for img_path in files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ Skipping unreadable image: {img_path.name}")
            continue

        faces = app.get(img)
        if not faces:
            # keep quiet-ish to avoid huge logs
            continue

        face = max(faces, key=face_area)
        area = face_area(face)

        # zoom-rescue pass
        if area < min_area:
            x1, y1, x2, y2 = map(int, face.bbox)
            pad = int(0.6 * max(x2 - x1, y2 - y1))

            h, w = img.shape[:2]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            crop = img[max(0, cy - pad):min(h, cy + pad), max(0, cx - pad):min(w, cx + pad)]
            if crop.size > 0:
                faces_zoom = app.get(crop)
                if faces_zoom:
                    face2 = max(faces_zoom, key=face_area)
                    area2 = face_area(face2)
                    if area2 > area:
                        face, area = face2, area2

        if area < min_area:
            continue

        data.append({"file": img_path.name, "embedding": face.normed_embedding})
        print(f"✅ Cached face: {img_path.name} (area {area}, min {min_area})")

    # ✅ CRITICAL FIX: ALWAYS write a cache file (even empty)
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)

    if data:
        print(f"✅ Rebuilt cache for theme '{theme_name}' with {len(data)} faces -> {cache_path}")
    else:
        print(f"⚠️ No valid faces found for theme: {theme_name} (wrote EMPTY cache) -> {cache_path}")

# ==================================================
# ENSURE ALL THEME CACHES EXIST
# ==================================================
def ensure_all_theme_caches():
    if not THEMES_DIR.exists():
        print(f"⚠️ THEMES_DIR missing: {THEMES_DIR}")
        return

    for theme_dir in THEMES_DIR.iterdir():
        if not theme_dir.is_dir():
            continue
        theme_name = theme_dir.name
        cache_path = CACHE_DIR / f"{theme_name}.pkl"
        if not cache_path.exists():
            print(f"⚠️ Cache missing for '{theme_name}', building…")
            build_cache_for_theme(theme_name)

# ==================================================
# REBUILD SINGLE THEME CACHE
# ==================================================
def rebuild_single_theme_cache(theme_name: str):
    print(f"🧠 Rebuilding cache for single theme: {theme_name}")
    build_cache_for_theme(theme_name)
