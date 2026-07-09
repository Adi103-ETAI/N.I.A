# ─────────────────────────────────────────────────────────────────────────────
# N.I.A Dockerfile — ported from Hermes Agent's Dockerfile architecture
# ─────────────────────────────────────────────────────────────────────────────
# Multi-stage build:
#   1. uv source (Python dep resolver)
#   2. node source (Node 22 LTS for frontend build)
#   3. Runtime (Debian 13 trixie with system deps)
#
# Key patterns ported from Hermes:
#   - Layer-cached dependency install (manifests first, source later)
#   - Non-root runtime user (UID 10000, overridable via NIA_UID)
#   - Immutable install tree (/opt/nia) + mutable data volume (/opt/data)
#   - Entry point with argument routing (bare exec vs niaharness subcommand)
#   - PYTHONUNBUFFERED + PYTHONDONTWRITEBYTECODE for container correctness
#   - uv sync with --frozen for reproducible dep resolution
# ─────────────────────────────────────────────────────────────────────────────

# ─── Stage 1: uv binary ────────────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:python3.13-bookworm AS uv_source

# ─── Stage 2: Node 22 LTS ──────────────────────────────────────────────────
FROM node:22-bookworm-slim AS node_source

# ─── Stage 3: Runtime ──────────────────────────────────────────────────────
FROM python:3.13-slim

# Disable Python stdout buffering; don't write .pyc at runtime.
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies in one layer, clear APT cache.
# Ported from Hermes: includes git, ripgrep, ffmpeg, procps, openssh-client,
# docker-cli (for Docker-in-Docker terminal backend), ca-certificates, curl.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates curl iputils-ping python3 python-is-python3 \
    ripgrep ffmpeg gcc g++ make cmake python3-dev python3-venv \
    libffi-dev procps git openssh-client docker-cli xz-utils \
    tini && \
    rm -rf /var/lib/apt/lists/*

# ─── Non-root runtime user ─────────────────────────────────────────────────
# UID can be overridden via NIA_UID at runtime (see docker/entrypoint.sh).
RUN useradd -u 10000 -m -d /opt/data nia

# ─── Copy uv binary from source stage ──────────────────────────────────────
COPY --chmod=0755 --from=uv_source /usr/local/bin/uv /usr/local/bin/uvx /usr/local/bin/

# ─── Copy Node 22 LTS from source stage ────────────────────────────────────
COPY --chmod=0755 --from=node_source /usr/local/bin/node /usr/local/bin/
COPY --from=node_source /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm
COPY --from=node_source /usr/local/lib/node_modules/corepack /usr/local/lib/node_modules/corepack
RUN ln -sf /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm && \
    ln -sf /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx && \
    ln -sf /usr/local/lib/node_modules/corepack/dist/corepack.js /usr/local/bin/corepack

# ─── Workdir ───────────────────────────────────────────────────────────────
WORKDIR /opt/nia

# ─── Layer-cached Python dependency install ────────────────────────────────
# Copy only pyproject.toml + uv.lock so the Python dep resolve + wheel
# download layer is cached unless those inputs change.
COPY pyproject.toml uv.lock ./
RUN touch ./README.md && \
    uv sync --frozen --no-install-project

# ─── Layer-cached frontend dependency install ──────────────────────────────
COPY frontend/terminal/package.json frontend/terminal/package-lock.json* frontend/terminal/
RUN cd frontend/terminal && npm install --prefer-offline --no-audit && npm cache clean --force

# ─── Source code ───────────────────────────────────────────────────────────
# .dockerignore excludes node_modules, .git, __pycache__, etc.
COPY --link --chmod=a+rX,go-w . .

# ─── Install NIA as editable (fast, no re-resolve) ─────────────────────────
RUN uv pip install --no-cache-dir --no-deps -e "."

# ─── Wire exec shim ────────────────────────────────────────────────────────
RUN mkdir -p /opt/nia/bin && \
    cp /opt/nia/docker/nia-exec-shim.sh /opt/nia/bin/niaharness && \
    chmod 0755 /opt/nia/bin/niaharness && \
    printf 'docker\n' > /opt/nia/.install_method

# ─── Runtime environment ───────────────────────────────────────────────────
ENV NIA_HOME=/opt/data
ENV PATH="/opt/nia/bin:/opt/nia/.venv/bin:/opt/data/.local/bin:${PATH}"
RUN mkdir -p /opt/data
VOLUME [ "/opt/data" ]

# tini as PID 1 for zombie reaping (MCP stdio subprocesses, git, etc.)
# Ported from Hermes Phase 1 (before s6-overlay migration).
ENTRYPOINT [ "tini", "--", "/opt/nia/docker/entrypoint.sh" ]
CMD [ ]
