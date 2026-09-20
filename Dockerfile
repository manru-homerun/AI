FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SHARED_GRU_ARTIFACT_DIR=/app/artifacts/shared_next_poi_gru_experiment/shared-next-poi-gru-v1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-production.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-production.txt

COPY src ./src
COPY artifacts/shared_next_poi_gru_experiment ./artifacts/shared_next_poi_gru_experiment

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/ready || exit 1

CMD ["uvicorn", "src.api.tiny_gru_app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
