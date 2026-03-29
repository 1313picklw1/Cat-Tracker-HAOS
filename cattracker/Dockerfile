ARG BUILD_FROM=python:3.12-slim-bookworm
FROM $BUILD_FROM

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    jq \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY cat_tracker/ ./cat_tracker/
COPY run.py requirements.txt gemini_cats.txt ./

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

COPY run.sh /
RUN chmod +x /run.sh

CMD ["/run.sh"]
