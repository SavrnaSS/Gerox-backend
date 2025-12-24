FROM python:3.10-slim

# ===============================
# System dependencies (required for OpenCV / PIL / Torch)
# ===============================
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ===============================
# Working directory
# ===============================
WORKDIR /app

# ===============================
# Python dependencies
# ===============================
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ===============================
# Create folders for models
# ===============================
RUN mkdir -p /app/weights /app/gfpgan/weights

# ===============================
# Download AI model weights (DO NOT COMMIT TO GIT)
# ===============================

# GFPGAN
RUN wget -O /app/weights/GFPGANv1.4.pth \
    https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth

# Real-ESRGAN
RUN wget -O /app/weights/RealESRGAN_x4plus.pth \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

# GFPGAN parsing & detection models
RUN wget -O /app/gfpgan/weights/parsing_parsenet.pth \
    https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_parsenet.pth \
 && wget -O /app/gfpgan/weights/detection_Resnet50_Final.pth \
    https://github.com/xinntao/facexlib/releases/download/v0.2.0/detection_Resnet50_Final.pth

# ===============================
# Copy application code
# ===============================
COPY . .

# ===============================
# Railway / production port
# ===============================
EXPOSE 8000

# ===============================
# Start server (Railway injects $PORT)
# ===============================
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
