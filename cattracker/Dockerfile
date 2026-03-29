ARG BUILD_FROM=python:3.12-slim-bookworm
FROM ${BUILD_FROM}

# Faster, more reliable pip in constrained Supervisor builds
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    jq \
    libglib2.0-0 \
    libgomp1 \
    libjpeg62-turbo \
    libopenjp2-7 \
    libpng16-16 \
    libtiff6 \
    libwebp7 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY cat_tracker/ ./cat_tracker/
COPY run.py requirements.txt gemini_cats.txt ./

# venv avoids Debian PEP 668 "externally managed" issues
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU wheels first keeps install smaller and avoids some ARM pip resolver timeouts
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY run.sh /
RUN chmod +x /run.sh

CMD ["/run.sh"]
