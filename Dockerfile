# PulseIQ — single-container deployment.
# Serves the FastAPI backend and the frontend from one origin, which removes
# the CORS preflight and the HTTPS-page-calling-HTTP-localhost problem.
#
# Python 3.12, not 3.11: the pinned scikit-learn / numpy / scipy versions this
# project's models were built with publish wheels for 3.12 upward.
FROM python:3.12-slim

# libgomp1 is required by scikit-learn and torch on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# torch first, from the CPU-only index. The default PyPI wheel drags in ~2 GB of
# CUDA libraries this app never touches, which alone can exhaust a free build.
RUN pip install --no-cache-dir torch==2.13.0 \
    --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The app directory is read-only on most hosts and is replaced on every deploy,
# so the database, JWT key and uploads live here instead.
ENV PULSEIQ_STATE_DIR=/data \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1
RUN mkdir -p /data && chmod 777 /data

EXPOSE 8001

# Shell form so $PORT expands — Render injects it, and it is not 8001.
# One worker: the models and the held-out test set load into memory at startup,
# and a second worker would double that on a 512 MB instance for no benefit.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8001} --workers 1
