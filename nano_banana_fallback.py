import os
import uuid
import base64
from io import BytesIO
from PIL import Image

from google import genai
from google.genai import types

# ==================================================
# FEATURE FLAGS (Railway-safe defaults)
# ==================================================
NANO_BANANA_VERBOSE = os.getenv("NANO_BANANA_VERBOSE", "1") == "1"

# OFF by default -> prevents InsightFace import + model load
NANO_BANANA_GENDER_DETECT = os.getenv("NANO_BANANA_GENDER_DETECT", "0") == "1"

# OFF by default -> prevents theme_cache_builder rebuild + "✅ Cached face..." spam
NANO_BANANA_REBUILD_CACHE = os.getenv("NANO_BANANA_REBUILD_CACHE", "0") == "1"

# OFF by default -> prevents uploading generated images to R2
NANO_BANANA_UPLOAD_R2 = os.getenv("NANO_BANANA_UPLOAD_R2", "0") == "1"

# OFF by default -> prevents saving generated images into themes_root/<theme>/
NANO_BANANA_SAVE_LOCAL = os.getenv("NANO_BANANA_SAVE_LOCAL", "0") == "1"


def _log(*args):
    if NANO_BANANA_VERBOSE:
        print(*args)


# ==================================================
# OPTIONAL IMPORTS (only when enabled)
# ==================================================
if NANO_BANANA_GENDER_DETECT:
    import cv2
    import numpy as np
    from insightface.app import FaceAnalysis

if NANO_BANANA_UPLOAD_R2:
    import r2_storage

if NANO_BANANA_REBUILD_CACHE:
    from theme_cache_builder import rebuild_single_theme_cache
    from theme_matcher import clear_theme_cache


# ==================================================
# INSIGHTFACE (LAZY LOAD)
# ==================================================
_face_app = None


def get_face_app():
    """
    Only used if NANO_BANANA_GENDER_DETECT=1.
    """
    global _face_app
    if not NANO_BANANA_GENDER_DETECT:
        return None

    if _face_app is None:
        _log("🧠 Loading InsightFace (gender detection)...")
        _face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        _face_app.prepare(ctx_id=0, det_size=(640, 640))
    return _face_app


# ==================================================
# THEME NAME NORMALIZER
# ==================================================
def normalize_theme_name(name: str) -> str:
    return (
        (name or "")
        .strip()
        .lower()
        .replace(" ", "")
        .replace("-", "_")
    )


# ==================================================
# GLOBAL SUBJECT PROMPTS (IDENTITY / GENDER)
# ==================================================
SUBJECT_PROMPTS = {
    "male": (
        "The person is a young man with masculine facial structure and "
        "natural male proportions. No makeup. Natural skin texture. "
        "Face must closely match the provided identity reference."
    ),
    "female": (
        "The person is a young woman with feminine facial structure and "
        "natural female proportions. Light natural makeup only. "
        "Soft facial features. Face must closely match the provided identity reference."
    ),
    "neutral": (
        "Face must closely match the provided identity reference."
    ),
}


# ==================================================
# THEME SCENE PROMPTS
# ==================================================
THEME_SCENES = {
    "selfie_with_scarlett_johnson": (
        "Taking a close-up selfie with Scarlett Johansson on an active movie set. "
        "Selfie mode, front camera perspective, handheld phone. "
        "Both are standing very close to the camera, smiling softly. "
        "Behind them is a full cinematic production environment — large ARRI film camera, "
        "monitor screens, boom mics hanging above, bright studio lights, reflectors, "
        "crew members working in the background. "
        "Scarlett Johansson has light movie makeup with subtle action-scene dirt and "
        "minor bruises, wearing a film costume. "
        "Style tags: ultra realistic, cinematic portrait, behind-the-scenes, "
        "movie shoot ambience, 8k detail, natural lighting, shallow depth of field."
    ),
    "black_hoddie_portrait": (
        "Ultra-realistic indoor portrait of the person wearing a black hoodie "
        "with a red spider-web graphic printed on the chest. "
        "Relaxed, confident posture with one hand resting casually behind the neck, "
        "natural candid expression. "
        "Shot indoors in a cozy bedroom environment, background filled with a collage "
        "of colorful posters and art prints on the wall. "
        "Soft cinematic indoor lighting with gentle shadows, realistic skin texture, "
        "natural color grading. "
        "Shallow depth of field with a slightly blurred background and sharp focus on the face. "
        "DSLR camera look, eye-level angle, 50mm lens feel, high dynamic range, "
        "ultra-detailed, photorealistic, modern lifestyle portrait, no over-stylization."
    ),
    "mountian_photoshoot": (
        "Highly realistic lifestyle portrait of the person sitting casually on a concrete "
        "roadside barrier in a mountainous environment. "
        "Wearing a light blue open button-down shirt layered over a plain white t-shirt, "
        "paired with black pants and clean white sneakers. "
        "Relaxed seated pose with legs slightly apart, one elbow resting on the knee and "
        "hand thoughtfully touching the chin, conveying a calm, confident, introspective vibe. "
        "Accessories include a gold wristwatch and a thin gold bracelet. "
        "Background features lush green hills and valleys with layered mountains fading into "
        "the distance beneath a bright, softly clouded sky. "
        "Natural daylight with soft shadows, cinematic depth of field, and realistic skin texture. "
        "Shot with a DSLR camera, eye-level angle, centered composition, shallow depth of field, "
        "ultra-detailed, natural color grading, candid travel lifestyle photography, "
        "high realism, 4K quality."
    ),
    "photoshoot_with_bike": (
        "Ultra-realistic cinematic outdoor lifestyle portrait of the person sitting confidently "
        "on a vivid green Kawasaki Ninja ZX-4R sports motorcycle on a quiet asphalt road. "
        "Wearing a bright green and black Kawasaki racing jacket with realistic sponsor patches, "
        "a plain black t-shirt underneath, black cargo pants, and clean white sneakers. "
        "The subject is seated upright on the motorcycle with a relaxed yet confident posture, "
        "hands resting naturally near the handlebars and legs grounded for balance, "
        "bike centered prominently in the frame. "
        "The motorcycle features aggressive angular fairings, visible front disc brake, "
        "racing decals, aerodynamic mirrors, and realistic reflections across the windshield "
        "and body panels, with headlights turned on emitting a subtle glow. "
        "Environment is a natural outdoor setting with lush green trees softly blurred in the "
        "background, shallow depth of field, clean road surface, and no traffic or distractions. "
        "Natural daylight with soft shadows, balanced highlights, and realistic skin texture. "
        "Shot using DSLR photography at eye-level angle with centered composition, "
        "cinematic color grading, bokeh background, ultra-sharp focus on both subject and bike. "
        "High realism, ultra-detailed, 4K clarity, lifestyle automotive photography style, "
        "Instagram-ready, no motion blur, no distortion, no visual artifacts."
    ),
    "selfie_with_cristiano_ronaldo": (
        "Ultra-realistic candid selfie scene on a professional football ground, "
        "captured from a handheld smartphone at arm’s length in selfie mode. "
        "The person is standing beside Cristiano Ronaldo, smiling naturally while "
        "taking a spontaneous fan selfie after a match. "
        "Cristiano Ronaldo is wearing a modern football training kit, with natural skin texture, "
        "realistic facial details, subtle sweat on the face, authentic hairstyle, "
        "and light beard stubble. "
        "Background features green stadium grass, a visible goalpost, and softly blurred "
        "stadium lights and crowd atmosphere, creating a realistic post-match environment. "
        "Natural daylight with soft shadows, realistic lighting, true-to-life colors, "
        "and accurate human proportions. "
        "Shallow depth of field, DSLR-level photo realism despite smartphone perspective, "
        "real skin pores, cinematic realism. "
        "The image should look like a genuine spontaneous fan selfie, not posed or staged. "
        "Extremely photorealistic, ultra-detailed, 8K quality, no distortion, "
        "no extra limbs, no blur, no waxy skin, no AI artifacts."
    ),
}

DEFAULT_PROMPT = (
    "Ultra realistic portrait photograph. "
    "DSLR quality, natural lighting."
)

TARGET_WIDTH = 1080
TARGET_HEIGHT = 1350


# ==================================================
# CUSTOM ERROR
# ==================================================
class NanoBananaError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


# ==================================================
# ADVANCED GENDER DETECTION (OPTIONAL)
# ==================================================
def detect_gender(face_bytes: bytes):
    """
    If NANO_BANANA_GENDER_DETECT=0, returns neutral immediately (no InsightFace).
    """
    if not NANO_BANANA_GENDER_DETECT:
        return "neutral", 0.0

    img = cv2.imdecode(np.frombuffer(face_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return "neutral", 0.0

    app = get_face_app()
    if app is None:
        return "neutral", 0.0

    faces = app.get(img)
    if not faces:
        return "neutral", 0.0

    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )

    gender_raw = face.gender
    male_prob = 1.0 if gender_raw == 1 else 0.0
    female_prob = 1.0 if gender_raw == 0 else 0.0

    x1, y1, x2, y2 = map(int, face.bbox)
    face_h = y2 - y1

    hair_region = img[max(0, y1 - int(face_h * 0.9)): y1 + int(face_h * 0.15), x1:x2]
    beard_region = img[y1 + int(face_h * 0.55): y2, x1:x2]

    long_hair = False
    if hair_region.size > 0:
        gray = cv2.cvtColor(hair_region, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
        long_hair = (np.count_nonzero(mask) / mask.size) > 0.38

    beard_detected = False
    if beard_region.size > 0:
        gray = cv2.cvtColor(beard_region, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 85, 255, cv2.THRESH_BINARY_INV)
        beard_detected = (np.count_nonzero(mask) / mask.size) > 0.22

    if beard_detected:
        return "male", 0.9
    if long_hair:
        return "female", 0.8
    if male_prob > 0.8:
        return "male", male_prob
    if female_prob > 0.8:
        return "female", female_prob

    return "neutral", 0.5


# ==================================================
# PROMPT BUILDER
# ==================================================
def build_prompt(theme: str, gender: str) -> str:
    key = normalize_theme_name(theme)
    scene = THEME_SCENES.get(key)

    if not scene:
        _log(f"⚠️ No scene prompt found for theme: {key}")
        return DEFAULT_PROMPT

    _log(f"✅ Using scene prompt: {key}")
    subject = SUBJECT_PROMPTS.get(gender, SUBJECT_PROMPTS["neutral"])

    return f"""
Use the provided face image as identity reference.
Do not change facial identity.

{subject}

{scene}

Generate a highly realistic portrait photo.
Aspect ratio 4:5 (portrait).
Professional DSLR quality.
"""


# ==================================================
# GEMINI NANO BANANA GENERATOR
# ==================================================
def generate_with_nano_banana(
    face_bytes: bytes,
    original_face_bytes: bytes,
    theme_name: str,
    themes_root: str,
):
    gender, confidence = detect_gender(original_face_bytes)
    _log(f"👤 Gender detected: {gender} (confidence {confidence:.2f})")

    prompt = build_prompt(theme_name, gender)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise NanoBananaError("NO_API_KEY", "GEMINI_API_KEY is not set")

    client = genai.Client(api_key=api_key)
    model = "gemini-2.5-flash-image"

    # Gemini expects base64 string for inline_data.data in this SDK usage
    face_b64 = base64.b64encode(face_bytes).decode("utf-8")

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part(text=prompt),
                types.Part(
                    inline_data={
                        "mime_type": "image/png",
                        "data": face_b64,
                    }
                ),
            ],
        )
    ]

    config = types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        temperature=0.4,
        top_p=0.8,
        max_output_tokens=1024,
    )

    image_bytes = None

    try:
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config,
        ):
            if (
                chunk.candidates
                and chunk.candidates[0].content
                and chunk.candidates[0].content.parts
            ):
                part = chunk.candidates[0].content.parts[0]
                if part.inline_data and part.inline_data.data:
                    image_bytes = part.inline_data.data
                    break

        if not image_bytes:
            raise NanoBananaError("NO_IMAGE", "No image returned from Gemini")

    except NanoBananaError:
        raise
    except Exception as e:
        raise NanoBananaError("GEMINI_ERROR", str(e))

    # ==================================================
    # FORCE EXACT 1080 × 1350
    # ==================================================
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    w, h = img.size

    target_ratio = TARGET_WIDTH / TARGET_HEIGHT
    current_ratio = w / h

    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))

    img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)

    buffer = BytesIO()
    img.save(buffer, format="PNG", optimize=True)
    image_bytes = buffer.getvalue()

    # ==================================================
    # OPTIONAL: SAVE GENERATED IMAGE INTO LOCAL THEMES DIR
    # ==================================================
    theme_key = normalize_theme_name(theme_name)
    filename = f"gen_{uuid.uuid4().hex}.png"

    if NANO_BANANA_SAVE_LOCAL:
        theme_dir = os.path.join(themes_root, theme_key)
        os.makedirs(theme_dir, exist_ok=True)

        save_path = os.path.join(theme_dir, filename)
        with open(save_path, "wb") as f:
            f.write(image_bytes)

        _log(f"💾 Saved Nano Banana image (local cache) → {save_path}")
    else:
        _log("💾 Local save skipped (NANO_BANANA_SAVE_LOCAL=0)")

    # ==================================================
    # OPTIONAL: UPLOAD GENERATED IMAGE INTO R2
    # ==================================================
    if NANO_BANANA_UPLOAD_R2 and getattr(r2_storage, "R2_ENABLED", False):
        r2_key = f"themes/{theme_key}/{filename}"
        r2_storage.put_bytes(r2_key, image_bytes, content_type="image/png")
        _log(f"☁️ Uploaded Nano Banana image to R2 → {r2_key}")
    else:
        _log("☁️ R2 upload skipped (NANO_BANANA_UPLOAD_R2=0 or R2 disabled)")

    # ==================================================
    # OPTIONAL: SAFE CACHE REBUILD (DISABLED BY DEFAULT)
    # ==================================================
    if NANO_BANANA_REBUILD_CACHE:
        try:
            _log(f"🧠 Rebuilding cache for theme: {theme_key}")
            rebuild_single_theme_cache(theme_key)
            clear_theme_cache(theme_key)
        except Exception as e:
            _log(f"⚠️ Cache rebuild skipped for {theme_key}: {e}")
    else:
        _log("🧠 Cache rebuild skipped (NANO_BANANA_REBUILD_CACHE=0)")

    _log("🍌 Nano Banana → SUCCESS")

    return image_bytes, filename
