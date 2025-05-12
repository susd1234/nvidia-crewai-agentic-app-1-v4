# Use Python 3.11 as the base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    graphviz \
    graphviz-dev \
    pkg-config \
    gcc \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p workdir data db

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LITELLM_LOG_LEVEL=CRITICAL
ENV NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
ENV NVIDIA_CHAT_MODEL=llama-3.1-70b-instruct
ENV NVIDIA_EMBEDDING_MODEL=nemo-retriever-e5-large
ENV LOG_LEVEL=INFO
ENV MAX_RETRIES=3
ENV TIMEOUT=60

# Expose ports
EXPOSE 8000
EXPOSE 7860

# Command to run the application
CMD ["python", "run_app.py"] 