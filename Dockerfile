# SLANT v2.0 Backend — Render Deployment
# ==========================================
# RAM budget: 512 MB (Render free tier)
# Uses python:3.11-slim to minimize base image size

FROM python:3.11-slim

# System deps for lxml, psycopg2, trafilatura
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    libxml2-dev \
    libxslt1-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps BEFORE copying code (leverages Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model (50 MB — stored in image, not downloaded at runtime)
RUN python -m spacy download en_core_web_sm

# Download NLTK data
RUN python -c "import nltk; nltk.download('vader_lexicon', quiet=True)"

# Download TextBlob corpora
RUN python -c "import textblob; textblob.download_corpora()" || true

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check — Render uses this to verify the service is up
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Start command
# --workers 1: CRITICAL for 512 MB RAM (multiple workers = multiple model copies)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
