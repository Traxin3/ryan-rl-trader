# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim AS base
LABEL org.opencontainers.image.source="https://github.com/Traxin3/ryan-rl-trader"
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps
RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Copy requirements and install (CPU by default)
COPY requirements.txt ./
# Also copy backend API requirements
COPY ryan_dash/backend_requirements.txt ./ryan_dash/backend_requirements.txt
# Torch CPU by default; allow override at build time
ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install --index-url ${TORCH_INDEX_URL} torch \
    && pip install -r requirements.txt \
    && pip install -r ryan_dash/backend_requirements.txt

# Copy code
COPY . .

# Non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000 3000

# Default command prints help
CMD ["python", "main.py", "--help"]
