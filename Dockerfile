# ============================================================
# Dockerfile — Cat vs Dog Classifier API
# ============================================================
# Base: python:3.10-slim  (CPU inference only — no CUDA needed)
# Final image size: ~1.5 GB
#
# Build:
#   docker build -t cat-dog-classifier:v1 .
#
# Run:
#   docker run -p 8000:8000 cat-dog-classifier:v1
#
# Test:
#   curl http://localhost:8000/health
# ============================================================

FROM python:3.10-slim

# ── Metadata ─────────────────────────────────────────────────────────────────
LABEL maintainer="MLOps Assignment M2"
LABEL description="Cat vs Dog image classifier REST API"
LABEL version="1.0.0"

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install CPU-only PyTorch first (much smaller than CUDA build, ~800MB saved)
# Then install the rest from requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        torch==2.5.1+cpu \
        torchvision==0.20.1+cpu \
        torchaudio==2.5.1+cpu \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
        fastapi==0.111.0 \
        uvicorn==0.30.1 \
        python-multipart==0.0.9 \
        Pillow==10.3.0 \
        numpy==1.26.4 \
        PyYAML==6.0.1 \
        scikit-learn==1.5.0

# ── Copy application code ─────────────────────────────────────────────────────
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/best_model.pt ./models/best_model.pt

# ── Expose API port ───────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# ── Run ───────────────────────────────────────────────────────────────────────
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
