# TEE Inference Worker - Hardened Docker Image
# This Dockerfile creates a minimal, secure container for running inference
# inside a GCP Confidential VM with the following security properties:
#
# 1. Read-only filesystem (enforced at runtime with --read-only)
# 2. No shell access (bash/sh removed)
# 3. Minimal attack surface (only Python + required libraries)
# 4. Fixed dependencies (pinned versions)
# 5. Non-root user
# 6. Deterministic build (reproducible)

# ============================================================================
# Stage 1: Build stage
# ============================================================================

FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /build

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================================
# Stage 2: Runtime stage (minimal, hardened)
# ============================================================================

FROM python:3.11-slim

# Metadata
LABEL maintainer="tee-inference-team"
LABEL description="Hardened TEE inference worker with attestation"
LABEL security.attestation="enabled"
LABEL security.read-only="true"
LABEL security.no-shell="true"

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ /app/src/

# Create directories for models (will be mounted or baked in)
RUN mkdir -p /app/models

# Bake model files into the image
COPY models/ /app/models/

# Create non-root user
RUN useradd -r -u 1000 -m -d /home/inference -s /usr/sbin/nologin inference && \
    chown -R inference:inference /app

# CRITICAL SECURITY HARDENING (MUST BE AFTER ALL RUN COMMANDS)
# Removing /bin/sh earlier breaks subsequent RUN layers.
RUN rm -f /bin/bash /bin/sh /bin/dash /bin/rbash /bin/zsh || true && \
    rm -f /usr/bin/apt /usr/bin/apt-get /usr/bin/dpkg || true && \
    rm -f /usr/bin/wget /usr/bin/curl || true && \
    rm -f /usr/bin/vi /usr/bin/vim /usr/bin/nano || true && \
    rm -rf /tmp/* /var/tmp/* /var/cache/* /root/.cache

# Switch to non-root user
USER inference

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Deterministic Python execution
ENV PYTHONHASHSEED=0

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python3 -c "import httpx; httpx.get('http://localhost:8000/health', timeout=2.0)" || exit 1

# Expose inference port
EXPOSE 8000

# Fixed entrypoint (cannot be overridden without changing image)
ENTRYPOINT ["python3", "-m", "uvicorn", "src.inference_server:app"]

# Default arguments (can be overridden but entrypoint is fixed)
CMD ["--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# Build-time arguments for model hash (optional - can be set at runtime)
ARG MODEL_HASH=""
ENV MODEL_HASH=${MODEL_HASH}

# Note: The actual enforcement of security happens at docker run time:
#   docker run --read-only --no-new-privileges --security-opt=no-new-privileges
#
# This Dockerfile prepares the image, but the systemd unit enforces the runtime security.

