FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ===============================
# System deps (needed to build insightface extensions)
# ===============================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libgl1 \
    libglib2.0-0 \
    curl \
    wget \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ===============================
# Python deps
# ===============================
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# ===============================
# Install insightface (CPU build)
# ===============================
RUN pip install --no-cache-dir --no-build-isolation insightface==0.7.3

# ===============================
# Model directories
# ===============================
RUN mkdir -p /app/weights /app/gfpgan/weights

# ===============================
# Download models
# ===============================
RUN wget -O /app/weights/GFPGANv1.4.pth \
    https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth

RUN wget -O /app/weights/RealESRGAN_x4plus.pth \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

RUN wget -O /app/gfpgan/weights/parsing_parsenet.pth \
    https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_parsenet.pth \
 && wget -O /app/gfpgan/weights/detection_Resnet50_Final.pth \
    https://github.com/xinntao/facexlib/releases/download/v0.2.0/detection_Resnet50_Final.pth

# ===============================
# App code
# ===============================
COPY . .

# Railway provides PORT env var
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
