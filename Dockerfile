FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Installs from a full, hash-pinned lockfile (every transitive dependency) for
# this CPU architecture — see scripts/lock.sh. Runtime dependencies only by
# default; CI builds with INSTALL_DEV=true to get pytest/ruff/jupyter.
ARG INSTALL_DEV=false
COPY locks/ locks/
RUN if [ "$INSTALL_DEV" = "true" ]; then KIND=dev; else KIND=runtime; fi; \
    LOCK="locks/$KIND-$(uname -m).txt"; \
    if [ ! -f "$LOCK" ]; then echo "No lockfile $LOCK for this architecture; run scripts/lock.sh" >&2; exit 1; fi; \
    pip install --no-cache-dir --require-hashes -r "$LOCK"

# Copy the project (.dockerignore keeps .env, data/, mlruns/ and .git out of the image)
COPY . .

# Set PYTHONPATH so absolute imports work
ENV PYTHONPATH=/app

# Expose ports for FastAPI (8000), Streamlit (8501), MLflow (5000)
EXPOSE 8000 8501 5000
