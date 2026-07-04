FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole project
COPY . .

# Set PYTHONPATH so absolute imports work
ENV PYTHONPATH=/app

# Expose ports for FastAPI (8000), Streamlit (8501), MLflow (5000)
EXPOSE 8000 8501 5000
