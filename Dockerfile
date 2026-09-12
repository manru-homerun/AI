FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TINY_GRU_ARTIFACT_DIR=/app/artifacts/tiny_gru_onnx_experiment \
    COURSE_DECODER_ARTIFACT_DIR=/app/artifacts/conditional_gru_decoder_experiment

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-production.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-production.txt

COPY src ./src
COPY artifacts/tiny_gru_onnx_experiment ./artifacts/tiny_gru_onnx_experiment
COPY artifacts/conditional_gru_decoder_experiment ./artifacts/conditional_gru_decoder_experiment

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "src.api.tiny_gru_app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
