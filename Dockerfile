FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ src/
COPY frontend/ frontend/
COPY main.py .

# Install dependencies and build
RUN uv sync --frozen --no-dev
RUN uv pip install -e . --no-deps

# --- Runtime stage ---
FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app

# Install frontend dependencies
RUN cd frontend/terminal && npm install --no-fund --no-audit 2>/dev/null || true

# Set environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV NIA_HOME=/root/.nia

# Create NIA home
RUN mkdir -p /root/.nia

# Default entrypoint
ENTRYPOINT ["python", "-m", "niaharness"]
