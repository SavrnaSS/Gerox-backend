FROM python:3.10

# ===============================
# System dependencies (FULL)
# ===============================
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
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
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ===============================
# Create model directories
# ===============================
RUN mkdir -p /app/weights /app/gfpgan/weights

# ===============================
# Download AI models
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
# Copy app
# ===============================
COPY . .

# ===============================
# Run server
# ===============================
EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
