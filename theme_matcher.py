import os
import pickle
import numpy as np
import random

# In-memory cache
THEME_CACHE = {}

# -----------------------------
# LOAD THEME CACHE
# -----------------------------
def load_theme_cache(theme_name: str):
    if theme_name in THEME_CACHE:
        return THEME_CACHE[theme_name]

    cache_path = os.path.join("theme_cache", f"{theme_name}.pkl")
    if not os.path.exists(cache_path):
        raise Exception(f"Theme cache not found for theme: {theme_name}")

    with open(cache_path, "rb") as f:
        data = pickle.load(f)

    if not data:
        raise Exception(f"Theme cache is empty for theme: {theme_name}")

    THEME_CACHE[theme_name] = data
    return data


# -----------------------------
# NORMALIZE CACHE ENTRY
# -----------------------------
def _extract_file_and_embedding(item):
    if isinstance(item, dict):
        return item.get("file"), item.get("embedding")

    if isinstance(item, (tuple, list)) and len(item) == 2:
        return item[0], item[1]

    return None, None


# -----------------------------
# TOP-K RANDOM MATCHER
# -----------------------------
def pick_best_theme_image(
    user_embedding,
    theme_faces,
    top_k: int = 3
):
    scored = []

    for item in theme_faces:
        file, emb = _extract_file_and_embedding(item)
        if file is None or emb is None:
            continue

        score = float(np.dot(user_embedding, emb))
        scored.append({"file": file, "score": score})

    if not scored:
        raise Exception("No valid embeddings in theme cache")

    scored.sort(key=lambda x: x["score"], reverse=True)

    top_candidates = scored[:max(1, min(top_k, len(scored)))]
    chosen = random.choice(top_candidates)

    return chosen["file"], round(chosen["score"], 3)


# -----------------------------
# 🆕 CLEAR IN-MEMORY CACHE
# -----------------------------
def clear_theme_cache(theme_name: str):
    """
    Clears in-memory cache for a theme
    so updated cache is reloaded next time
    """
    if theme_name in THEME_CACHE:
        del THEME_CACHE[theme_name]
