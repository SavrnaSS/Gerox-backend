import os
import cv2
import torch
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

print("✨ Loading enhancer models")

BASE_DIR = os.path.dirname(__file__)
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

GFPGAN_PATH = os.path.join(WEIGHTS_DIR, "GFPGANv1.4.pth")
REALESRGAN_PATH = os.path.join(WEIGHTS_DIR, "RealESRGAN_x4plus.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== GFPGAN =====================
gfpgan = GFPGANer(
    model_path=GFPGAN_PATH,
    upscale=2,                 # 🔥 stronger restoration
    arch="clean",
    channel_multiplier=2,
    bg_upsampler=None,
    device=DEVICE,
)

# ===================== RealESRGAN =====================
rrdbnet = RRDBNet(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=4,
)

upscaler = RealESRGANer(
    scale=4,
    model_path=REALESRGAN_PATH,
    model=rrdbnet,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False,
    device=DEVICE,
)

# =====================================================
# PUBLIC FACE ENHANCER
# =====================================================
def enhance_face_crop(face_crop, strength=1.0):
    """
    strength: 0.0 – 1.0 (0.8–1.0 recommended)
    """

    h, w, _ = face_crop.shape

    # ⬆️ TEMP UPSCALE (IMPORTANT FOR DETAIL)
    face_crop = cv2.resize(
        face_crop,
        (w * 2, h * 2),
        interpolation=cv2.INTER_CUBIC
    )

    # 🧠 GFPGAN RESTORATION
    _, _, restored = gfpgan.enhance(
        face_crop,
        has_aligned=False,
        only_center_face=True,
        paste_back=False
    )

    if restored is None:
        restored = face_crop

    # 🔍 RealESRGAN DETAIL PASS
    restored, _ = upscaler.enhance(restored, outscale=1)

    # 🎨 SHARPNESS + MICRO DETAIL
    restored = cv2.detailEnhance(
        restored,
        sigma_s=12 * strength,
        sigma_r=0.18 * strength
    )

    restored = cv2.addWeighted(
        restored, 1.15,
        cv2.GaussianBlur(restored, (0, 0), 1.0), -0.15,
        0
    )

    # ⬇️ RESIZE BACK
    restored = cv2.resize(
        restored,
        (w, h),
        interpolation=cv2.INTER_CUBIC
    )

    return restored
