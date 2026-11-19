# Build from repository root:
# docker build -f deployments/model-serving.Dockerfile -t interview-model-service .

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN groupadd --system app && useradd --system --gid app --create-home app
WORKDIR /app

COPY examples/requirements.txt /tmp/requirements.txt
RUN python -m pip install --requirement /tmp/requirements.txt

COPY examples/ml_ds_demos/model_serving_api.py /app/model_serving_api.py

USER app
EXPOSE 8010
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8010/healthz')"

CMD ["uvicorn", "model_serving_api:app", "--app-dir", "/app", "--host", "0.0.0.0", "--port", "8010"]
