# Supervisor sets BUILD_FROM from build.yaml (must be Debian here — Alpine has no apt-get).
ARG BUILD_FROM
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

# HA base-debian is minimal: install Python 3.11 + venv + runtime libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
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

# amd64: smaller CPU wheels from PyTorch; aarch64 (Pi): full wheels from PyPI.
RUN pip install --upgrade pip setuptools wheel \
 && ARCH="$(uname -m)" \
 && if [ "$ARCH" = "x86_64" ]; then \
      pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision; \
    else \
      pip install --no-cache-dir torch torchvision; \
    fi \
 && pip install --no-cache-dir --prefer-binary -r requirements.txt

CMD ["/run.sh"]
