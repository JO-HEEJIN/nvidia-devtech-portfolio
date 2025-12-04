# Healthcare VLM Deployment - Production Dockerfile
# Multi-stage build optimized for medical AI inference
# Supports CUDA, TensorRT, and healthcare-specific requirements

# Stage 1: Build environment with development dependencies
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgoogle-glog-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip and install build tools
RUN python3 -m pip install --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install TensorRT (if not already included in base image)
RUN pip install --no-cache-dir \
    tensorrt \
    pycuda \
    && pip cache purge

# Stage 2: Runtime environment for production deployment
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 as runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TZ=UTC
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Healthcare-specific environment variables
ENV HEALTHCARE_API_MODE=production
ENV LOG_LEVEL=INFO
ENV HIPAA_COMPLIANCE=enabled
ENV MAX_BATCH_SIZE=32
ENV DEFAULT_BACKEND=auto

# Create non-root user for security
RUN groupadd -r healthcareai && useradd -r -g healthcareai -s /bin/false healthcareai

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.8/dist-packages /usr/local/lib/python3.8/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create application directories with proper permissions
RUN mkdir -p /app \
    /app/models \
    /app/onnx_models \
    /app/tensorrt_engines \
    /app/logs \
    /app/cache \
    /app/sample_data \
    && chown -R healthcareai:healthcareai /app

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=healthcareai:healthcareai . /app/

# Create necessary subdirectories for models and data
RUN mkdir -p \
    /app/checkpoints \
    /app/calibration_cache \
    /app/results \
    /app/benchmark_results \
    && chown -R healthcareai:healthcareai /app

# Install Python dependencies specific to the application
RUN pip install --no-cache-dir \
    uvicorn[standard]==0.24.0 \
    gunicorn==21.2.0 \
    && pip cache purge

# Copy health check script
COPY --chown=healthcareai:healthcareai docker/healthcheck.py /app/healthcheck.py
RUN chmod +x /app/healthcheck.py

# Copy entrypoint script
COPY --chown=healthcareai:healthcareai docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose port for FastAPI application
EXPOSE 8000

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 /app/healthcheck.py || exit 1

# Set volume for persistent data
VOLUME ["/app/models", "/app/logs", "/app/cache"]

# Switch to non-root user
USER healthcareai

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# Labels for metadata and healthcare compliance
LABEL maintainer="Healthcare AI Team" \
      version="1.0.0" \
      description="Healthcare Vision-Language Model API" \
      healthcare.compliance="HIPAA" \
      healthcare.domain="medical_imaging" \
      healthcare.model="BiomedCLIP" \
      license="Healthcare Institution License" \
      org.opencontainers.image.title="Healthcare VLM API" \
      org.opencontainers.image.description="Production-ready medical imaging API with BiomedCLIP" \
      org.opencontainers.image.vendor="Healthcare Institution" \
      org.opencontainers.image.licenses="Healthcare License" \
      org.opencontainers.image.source="https://github.com/healthcare/vlm-deployment"

# Security and optimization notes:
# 1. Multi-stage build reduces final image size
# 2. Non-root user for security
# 3. Minimal runtime dependencies  
# 4. Health checks for container orchestration
# 5. Proper volume mounts for persistent data
# 6. CUDA runtime optimized for inference
# 7. Healthcare-specific environment variables
# 8. HIPAA compliance considerations built-in