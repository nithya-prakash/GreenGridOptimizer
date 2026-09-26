FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Runtime dependencies only by default; CI builds with INSTALL_DEV=true to get
# pytest/ruff (see requirements-dev.txt).
ARG INSTALL_DEV=false
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$INSTALL_DEV" = "true" ]; then pip install --no-cache-dir -r requirements-dev.txt; fi

# Copy the project (.dockerignore keeps .env, data/, mlruns/ and .git out of the image)
COPY . .

# Set PYTHONPATH so absolute imports work
ENV PYTHONPATH=/app

# Expose ports for FastAPI (8000), Streamlit (8501), MLflow (5000)
EXPOSE 8000 8501 5000
