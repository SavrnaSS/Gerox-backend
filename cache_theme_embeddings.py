# cache_theme_embeddings.py

import os
import cv2
import pickle
from insightface.app import FaceAnalysis
from tqdm import tqdm

THEMES_DIR = "public/themes"
CACHE_DIR = "theme_cache"
MIN_FACE = 40

os.makedirs(CACHE_DIR, exist_ok=True)

print("🚀 Initializing InsightFace")
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))


def is_image(f):
    return f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))


def extract_embedding(path):
    img = cv2.imread(path)
    if img is None:
        return None

    faces = app.get(img)
    if not faces:
        return None

    valid = []
    for f in faces:
        x1, y1, x2, y2 = map(int, f.bbox)
        if (x2 - x1) > MIN_FACE and (y2 - y1) > MIN_FACE:
            valid.append(f)

    if not valid:
        return None

    face = max(valid, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    return face.normed_embedding


for theme in os.listdir(THEMES_DIR):
    theme_path = os.path.join(THEMES_DIR, theme)
    if not os.path.isdir(theme_path):
        continue

    print(f"\n🎨 Processing theme: {theme}")
    data = []

    for img in tqdm(os.listdir(theme_path)):
        if not is_image(img):
            continue

        emb = extract_embedding(os.path.join(theme_path, img))
        if emb is not None:
            data.append((img, emb))

    if not data:
        print("❌ No faces found")
        continue

    with open(os.path.join(CACHE_DIR, f"{theme}.pkl"), "wb") as f:
        pickle.dump(data, f)

    print(f"✅ Cached {len(data)} faces")
