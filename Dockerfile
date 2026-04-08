FROM python:3.11-slim

# Hugging Face Spaces metadata
LABEL maintainer="hackathon-team"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
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
RUN python -c "\
try:\
    from datasets import load_dataset;\
    load_dataset('rajpurkar/squad', split='validation[:200]');\
    print('SQuAD cached.');\
    load_dataset('allenai/qasper', split='validation', trust_remote_code=True);\
    print('QASPER cached.');\
except Exception as e:\
    print(f'Dataset cache skipped: {e}');\
" || true

# Hugging Face Spaces uses port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/health').raise_for_status()" || exit 1

# Start server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
