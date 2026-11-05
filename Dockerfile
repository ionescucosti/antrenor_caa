# Build context: the repository root. Serves the Rokid AI RAG Backend (rokid_backend).
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-ron curl \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 10001 rokid
WORKDIR /srv
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY rokid_backend ./rokid_backend
COPY app.py engine.py importer.py ./
COPY static ./static
RUN mkdir -p /data/caa && chown -R rokid:rokid /data /srv
USER rokid
ENV ROKID_DATA_DIR=/data ROKID_CAA_DATA_DIR=/data/caa TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 CMD curl -fsS http://127.0.0.1:8000/api/v1/health || exit 1
CMD ["uvicorn", "rokid_backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
