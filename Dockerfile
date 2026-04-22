# HF Spaces Dockerfile for Spaces Pipeline Pro
# Serves the OpenEnv HTTP contract on port 8000.
FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Copy env code
COPY . /app

# Install package + runtime deps
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir uvicorn fastapi

# HF Spaces expects the app to bind on 0.0.0.0 at app_port (8000 per README front-matter)
ENV PYTHONUNBUFFERED=1 \
    SPACES_MODE=mock \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000

# Simple healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://localhost:8000/ || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
