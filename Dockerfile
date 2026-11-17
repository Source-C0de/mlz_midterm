FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    MODEL_PATH=/app/artifacts/model.joblib

WORKDIR /app

# Dependencies first so the layer is cached across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code.
COPY train.py predict.py ./
COPY src/ ./src/

# Train the model at build time so the image ships a ready-to-serve artifact.
# Training is deterministic (fixed seed inside train.py), so rebuilds produce
# the same model. Pass --fast for a quicker build.
COPY data/raw/ai4i2020.csv ./data/raw/ai4i2020.csv
RUN python train.py --data data/raw/ai4i2020.csv --out-dir artifacts

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--threads", "4", \
     "--timeout", "60", "--access-logfile", "-", "predict:app"]
