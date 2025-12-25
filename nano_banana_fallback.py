# nano_banana_fallback.py
import os
import re
import time
import io
from typing import Tuple

from PIL import Image
from google import genai


class NanoBananaError(Exception):
    pass


# Lazy singleton client (avoids re-init per request)
_genai_client = None

def _client():
    global _genai_client
    if _genai_client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise NanoBananaError("GEMINI_API_KEY is not set")
        _genai_client = genai.Client(api_key=api_key)
    return _genai_client


def _normalize_theme_name(name: str) -> str:
    return (
        (name or "")
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def _theme_prompt(theme_key: str) -> str:
    """
    Main prompt templates. Add more themes here anytime.
    IMPORTANT: We do NOT use theme images anymore.
    """
    prompts = {
        "selfie_with_scarlett_johnson": (
            "Create an ultra-realistic smartphone selfie of the user standing next to Scarlett Johansson. "
            "Keep the user's face identity strongly preserved. Natural skin texture, correct facial proportions, "
            "cinematic soft lighting, shallow depth of field, modern casual indoor background. "
            "Make it look like a real photo, not a cartoon. No text, no watermark."
        ),
        "selfie_with_cristiano_ronaldo": (
            "Create an ultra-realistic selfie of the user with Cristiano Ronaldo. "
            "Keep the user's face identity strongly preserved. Bright natural lighting, realistic smartphone perspective, "
            "sharp focus on faces, soft background bokeh. No text, no watermark."
        ),
        "black_hoddie_portrait": (
            "Create a high-quality portrait photo of the user wearing a black hoodie. "
            "Keep the user's face identity strongly preserved. Moody studio lighting, realistic fabric texture, "
            "clean background, DSLR look, no text, no watermark."
        ),
        "default": (
            "Create a high-quality photorealistic image of the user in a stylish modern portrait. "
            "Keep the user's face identity strongly preserved. No text, no watermark."
        ),
    }

    if theme_key in prompts:
        return prompts[theme_key]

    # Generic fallback: turn theme slug into a readable hint
    readable = re.sub(r"[_]+", " ", theme_key).strip()
    return (
        f"Create a photorealistic image of the user in the theme: '{readable}'. "
        "Keep the user's face identity strongly preserved. Realistic lighting and textures. "
        "No text, no watermark."
    )


def _to_png_bytes(img_bytes: bytes) -> bytes:
    """
    Gemini may return JPG bytes. Convert to PNG bytes for consistency.
    """
    try:
        im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        out = io.BytesIO()
        im.save(out, format="PNG", optimize=True)
        return out.getvalue()
    except Exception:
        # if conversion fails, return raw
        return img_bytes


def generate_with_nano_banana(
    *,
    face_bytes: bytes,
    original_face_bytes: bytes,  # kept for signature compatibility (unused)
    theme_name: str,
    themes_root: str,  # kept for signature compatibility (unused)
) -> Tuple[bytes, str]:
    """
    Returns: (png_bytes, filename_label)
    This now works WITHOUT theme images/caches.
    """
    theme_key = _normalize_theme_name(theme_name) or "default"
    prompt = _theme_prompt(theme_key)

    # Send face as conditioning image (identity anchor)
    contents = [
        {
            "role": "user",
            "parts": [
                {"text": prompt[:2500]},
                {"inline_data": {"mime_type": "image/jpeg", "data": face_bytes}},
            ],
        }
    ]

    try:
        client = _client()

        output_image = None
        for chunk in client.models.generate_content_stream(
            model=os.getenv("NANO_BANANA_MODEL", "gemini-2.5-flash-image"),
            contents=contents,
        ):
            if not getattr(chunk, "candidates", None):
                continue
            for candidate in chunk.candidates:
                content = getattr(candidate, "content", None)
                parts = getattr(content, "parts", None) if content else None
                if not parts:
                    continue
                for part in parts:
                    inline = getattr(part, "inline_data", None)
                    if inline and getattr(inline, "data", None):
                        output_image = inline.data

        if not output_image:
            raise NanoBananaError("No image returned by Gemini (nano banana)")

        png = _to_png_bytes(output_image)
        fname = f"nano_{theme_key}_{time.strftime('%Y%m%d_%H%M%S')}.png"
        return png, fname

    except NanoBananaError:
        raise
    except Exception as e:
        raise NanoBananaError(str(e))
