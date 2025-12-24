FROM python:3.10-slim

# ---- system dependencies (CRITICAL) ----
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---- working dir ----
WORKDIR /app

# ---- python deps ----
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---- copy app ----
COPY . .

# ---- expose port ----
EXPOSE 8000

# ---- start ----
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
