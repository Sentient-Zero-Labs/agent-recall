FROM python:3.11-slim

WORKDIR /app

# Install build deps then clean up in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy package metadata first — lets Docker cache the pip install layer
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install the package (no dev/embedding extras in prod image)
RUN pip install --no-cache-dir .

# Data directory for the SQLite database
RUN mkdir -p /data

ENV RECALL_DB_PATH=/data/recall.db

EXPOSE 8678

CMD ["recall", "serve", "--host", "0.0.0.0", "--port", "8678", "--db", "/data/recall.db"]
