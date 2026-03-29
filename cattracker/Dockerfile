# Home Assistant injects BUILD_FROM, BUILD_ARCH, BUILD_VERSION when building.
ARG BUILD_FROM=python:3.12-slim-bookworm
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

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    jq \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY cat_tracker/ ./cat_tracker/
COPY run.py requirements.txt gemini_cats.txt ./
COPY run.sh /run.sh
RUN chmod +x /run.sh

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Arch-aware torch: /whl/cpu is for amd64 only; aarch64 uses PyPI wheels (Pi-friendly).
RUN pip install --upgrade pip setuptools wheel \
 && ARCH="$(uname -m)" \
 && if [ "$ARCH" = "x86_64" ]; then \
      pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision; \
    else \
      pip install --no-cache-dir torch torchvision; \
    fi \
 && pip install --no-cache-dir --prefer-binary -r requirements.txt

CMD ["/run.sh"]
