# ─────────────────────────────────────────────────────────────
# Infant Growth Digital Twin v3.0  —  Multi-Method ML + WHO
# Docker image for Render cloud deployment
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/raw data/models

# Train models at build time (downloads NHANES + trains 8 ML methods)
# This ensures the Docker image ships with pre-trained models
RUN python -m src.training.train

# Expose port
EXPOSE 10000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:10000/health || exit 1

# Run the API server
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "10000"]
