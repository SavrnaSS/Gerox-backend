FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ===============================
# System deps (opencv/insightface runtime + build tools)
# ===============================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    git \
    curl \
    wget \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ===============================
# Python deps
# ===============================
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# ===============================
# (Optional) model dirs
# ===============================
RUN mkdir -p /app/weights /app/gfpgan/weights

# ===============================
# (Optional) download big models
# Default: SKIP (better for Railway)
# ===============================
ARG DOWNLOAD_MODELS=0
RUN if [ "$DOWNLOAD_MODELS" = "1" ]; then \
      echo "Downloading GFPGAN/RealESRGAN models..." && \
      wget -q -O /app/weights/GFPGANv1.4.pth \
        https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth && \
      wget -q -O /app/weights/RealESRGAN_x4plus.pth \
        https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth && \
      wget -q -O /app/gfpgan/weights/parsing_parsenet.pth \
        https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_parsenet.pth && \
      wget -q -O /app/gfpgan/weights/detection_Resnet50_Final.pth \
        https://github.com/xinntao/facexlib/releases/download/v0.2.0/detection_Resnet50_Final.pth ; \
    else \
      echo "Skipping model downloads (DOWNLOAD_MODELS=0)"; \
    fi

# ===============================
# App code
# ===============================
COPY . .

# Railway provides PORT env var
CMD ["bash", "-lc", "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
