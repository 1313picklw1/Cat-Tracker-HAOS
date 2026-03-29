# Debian bookworm via official Python image (Docker Hub). Supervisor passes BUILD_FROM from build.yaml.
ARG BUILD_FROM=python:3.11-slim-bookworm
FROM ${BUILD_FROM}

ARG BUILD_ARCH=unknown
ARG BUILD_VERSION=0

LABEL \
    io.hass.type="addon" \
    io.hass.arch="${BUILD_ARCH}" \
    io.hass.version="${BUILD_VERSION}"

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Base image already includes Python + pip; only add OS tools for OpenCV/ffmpeg and jq for run.sh.
RUN echo "=== CatTracker [1/6] apt-get: ffmpeg, jq, libs ===" && date -u \
 && apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    jq \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
 && echo "=== CatTracker [1/6] done ===" && date -u

WORKDIR /app

COPY cat_tracker/ ./cat_tracker/
COPY run.py requirements.txt gemini_cats.txt ./
COPY run.sh /run.sh
RUN chmod +x /run.sh

RUN echo "=== CatTracker [2/6] python venv ===" && date -u \
 && python -m venv /opt/venv \
 && echo "=== CatTracker [2/6] done ===" && date -u

ENV PATH="/opt/venv/bin:$PATH"

RUN echo "=== CatTracker [3/6] pip toolchain (short) ===" && date -u \
 && pip install --upgrade pip setuptools wheel \
 && echo "=== CatTracker [3/6] done ===" && date -u

# This step downloads hundreds of MB — often 10–40 minutes on slow links; UI stays on "Installing".
RUN echo "=== CatTracker [4/6] PyTorch + torchvision (LARGE download, please wait) ===" && date -u \
 && ARCH="$(uname -m)" \
 && if [ "$ARCH" = "x86_64" ]; then \
      pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision; \
    else \
      pip install --no-cache-dir torch torchvision; \
    fi \
 && echo "=== CatTracker [4/6] done ===" && date -u

RUN echo "=== CatTracker [5/6] ultralytics + opencv + flask + gemini + mqtt ===" && date -u \
 && pip install --no-cache-dir --prefer-binary -r requirements.txt \
 && echo "=== CatTracker [5/6] done ===" && date -u

RUN echo "=== CatTracker [6/6] image ready ===" && date -u

CMD ["/run.sh"]
