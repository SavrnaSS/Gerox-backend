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
# App code
# ===============================
COPY . .

# Railway provides PORT env var
CMD ["bash", "-lc", "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
