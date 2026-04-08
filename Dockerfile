FROM python:3.11-slim

# Hugging Face Spaces metadata
LABEL maintainer="hackathon-team"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    socat \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY models.py .
COPY client.py .
COPY openenv.yaml .
COPY graders/ ./graders/
COPY tasks/ ./tasks/
COPY server/ ./server/
COPY inference.py .

# Pre-download datasets at build time to avoid cold-start delays
# (Falls back to hardcoded samples if download fails)
RUN python - <<'PY' || true
try:
    from datasets import load_dataset
    load_dataset("rajpurkar/squad", split="validation[:200]")
    print("SQuAD cached.")
except Exception as e:
    print(f"SQuAD cache skipped: {e}")
PY

# Hugging Face Spaces commonly uses 7860, while local OpenEnv docker providers
# often inject PORT=8000. Support both.
EXPOSE 7860
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import os, requests; port=os.environ.get('PORT', '7860'); requests.get(f'http://localhost:{port}/health').raise_for_status()" || exit 1

# Start server on 7860 and mirror traffic from 8000 for local OpenEnv docker clients.
CMD ["sh", "-c", "socat TCP-LISTEN:8000,fork,reuseaddr TCP:127.0.0.1:${PORT:-7860} & python -m server.app"]
