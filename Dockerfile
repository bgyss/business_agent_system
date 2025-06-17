# Multi-stage Dockerfile for Business Agent System
FROM nixos/nix:latest AS nix-builder

# Enable flakes
RUN echo "experimental-features = nix-command flakes" >> /etc/nix/nix.conf

# Copy source
WORKDIR /app
COPY . .

# Build with Nix
RUN nix build

# Extract the built package
RUN cp -r result/ /output/

# Runtime stage
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    sqlite3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1000 appuser

# Copy built application from nix-builder
COPY --from=nix-builder /output /app
COPY --chown=appuser:appuser . /app/src

# Set working directory
WORKDIR /app/src

# Install uv
RUN pip install uv

# Switch to app user
USER appuser

# Install Python dependencies
RUN uv sync --frozen

# Create necessary directories
RUN mkdir -p logs

# Environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 8501

# Default command
CMD ["uv", "run", "python", "main.py", "--config", "config/restaurant_config.yaml"]

# Alternative: Use Nix-built binary directly
# FROM nixos/nix:latest
# COPY --from=nix-builder /output /app
# WORKDIR /app
# CMD ["./bin/business-agent-system"]