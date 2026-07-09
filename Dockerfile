# ─────────────────────────────────────────────────────────────────────────────
# N.I.A Dockerfile — ported from Hermes Agent's Dockerfile
# Includes: s6-overlay process supervision, Playwright, lazy-install,
# 3-script entrypoint, multi-stage build, non-root runtime user.
# ─────────────────────────────────────────────────────────────────────────────

# ─── Stage 1: uv binary ────────────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:python3.13-bookworm AS uv_source

# ─── Stage 2: Node 22 LTS (Debian bookworm-slim for glibc compat) ──────────
FROM node:22-bookworm-slim AS node_source

# ─── Stage 3: Runtime ──────────────────────────────────────────────────────
FROM debian:13

# Disable Python stdout buffering; don't write .pyc at runtime.
# Ported from Hermes: /opt/nia is immutable at runtime, writable state
# belongs under /opt/data (the VOLUME).
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Store Playwright browsers outside the volume mount so the build-time
# install survives the /opt/data volume overlay at runtime.
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/nia/.playwright

# Install system dependencies in one layer, clear APT cache.
# Ported from Hermes: includes git, ripgrep, ffmpeg, docker-cli (for
# DooD terminal backend), openssh-client, procps, tini.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates curl iputils-ping python3 python-is-python3 \
    ripgrep ffmpeg gcc g++ make cmake python3-dev python3-venv \
    libffi-dev libolm-dev procps git openssh-client docker-cli xz-utils && \
    rm -rf /var/lib/apt/lists/*

# ─── s6-overlay install ────────────────────────────────────────────────────
# Ported from Hermes Dockerfile lines 34-89. s6-overlay provides:
#   - PID 1 process supervision (/init replaces tini)
#   - cont-init.d scripts (run as root before services start)
#   - s6-rc service management (longrun services with dependencies)
#   - Non-blocking zombie reaping (SIGCHLD handler in s6-svscan)
#
# Multi-arch: BuildKit auto-populates TARGETARCH (amd64 / arm64).
# Supply-chain integrity: every tarball is checksum-verified.
ARG TARGETARCH
ARG S6_OVERLAY_VERSION=3.2.3.0
ARG S6_OVERLAY_NOARCH_SHA256=b720f9d9340efc8bb07528b9743813c836e4b02f8693d90241f047998b4c53cf
ARG S6_OVERLAY_X86_64_SHA256=a93f02882c6ed46b21e7adb5c0add86154f01236c93cd82c7d682722e8840563
ARG S6_OVERLAY_AARCH64_SHA256=0952056ff913482163cc30e35b2e944b507ba1025d78f5becbb89367bf344581
ARG S6_OVERLAY_SYMLINKS_SHA256=a60dc5235de3ecbcf874b9c1f18d73263ab99b289b9329aa950e8729c4789f0e
ADD https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}/s6-overlay-noarch.tar.xz /tmp/
ADD https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}/s6-overlay-symlinks-noarch.tar.xz /tmp/
RUN set -eu; \
    case "${TARGETARCH:-amd64}" in \
        amd64) s6_arch="x86_64"; s6_arch_sha="${S6_OVERLAY_X86_64_SHA256}" ;; \
        arm64) s6_arch="aarch64"; s6_arch_sha="${S6_OVERLAY_AARCH64_SHA256}" ;; \
        *) echo "Unsupported TARGETARCH=${TARGETARCH} for s6-overlay" >&2; exit 1 ;; \
    esac; \
    curl -fsSL --retry 3 -o /tmp/s6-overlay-arch.tar.xz \
        "https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}/s6-overlay-${s6_arch}.tar.xz"; \
    { \
        printf '%s  %s\n' "${S6_OVERLAY_NOARCH_SHA256}" /tmp/s6-overlay-noarch.tar.xz; \
        printf '%s  %s\n' "${s6_arch_sha}" /tmp/s6-overlay-arch.tar.xz; \
        printf '%s  %s\n' "${S6_OVERLAY_SYMLINKS_SHA256}" /tmp/s6-overlay-symlinks-noarch.tar.xz; \
    } > /tmp/s6-overlay.sha256; \
    sha256sum -c /tmp/s6-overlay.sha256; \
    tar -C / -Jxpf /tmp/s6-overlay-noarch.tar.xz; \
    tar -C / -Jxpf /tmp/s6-overlay-arch.tar.xz; \
    tar -C / -Jxpf /tmp/s6-overlay-symlinks-noarch.tar.xz; \
    rm /tmp/s6-overlay-*.tar.xz /tmp/s6-overlay.sha256; \
    # Backward-compat shim: symlink /usr/bin/tini → /init so external
    # wrappers that hard-code tini still work.
    ln -sf /init /usr/bin/tini

# ─── Non-root runtime user ─────────────────────────────────────────────────
# UID can be overridden via NIA_UID at runtime (see docker/stage2-hook.sh).
RUN useradd -u 10000 -m -d /opt/data nia

# ─── Copy uv binary from source stage ──────────────────────────────────────
COPY --chmod=0755 --from=uv_source /usr/local/bin/uv /usr/local/bin/uvx /usr/local/bin/

# ─── Copy Node 22 LTS from source stage ────────────────────────────────────
# Ported from Hermes: copies node + npm + corepack binaries, recreates
# symlinks. Debian 13's bundled nodejs is pinned to 20.x which reached
# EOL — we use Node 22 LTS instead.
COPY --chmod=0755 --from=node_source /usr/local/bin/node /usr/local/bin/
COPY --from=node_source /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm
COPY --from=node_source /usr/local/lib/node_modules/corepack /usr/local/lib/node_modules/corepack
RUN ln -sf /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm && \
    ln -sf /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx && \
    ln -sf /usr/local/lib/node_modules/corepack/dist/corepack.js /usr/local/bin/corepack

# ─── Workdir ───────────────────────────────────────────────────────────────
WORKDIR /opt/nia

# ─── Layer-cached frontend dependency install ──────────────────────────────
# Copy only package manifests first so npm install is cached unless
# the lockfiles change.
COPY frontend/terminal/package.json frontend/terminal/package-lock.json* frontend/terminal/

ENV npm_config_install_links=false
RUN cd frontend/terminal && npm install --prefer-offline --no-audit && \
    npm cache clean --force

# ─── Layer-cached Python dependency install ────────────────────────────────
# Copy only pyproject.toml + uv.lock so the Python dep resolve + wheel
# download layer is cached unless those inputs change.
COPY pyproject.toml uv.lock ./
RUN touch ./README.md && \
    uv sync --frozen --no-install-project

# ─── Frontend build (cached independently from Python source) ──────────────
COPY frontend/terminal/ frontend/terminal/
RUN cd frontend/terminal && npx tsc --noEmit || true

# ─── Playwright browser install ────────────────────────────────────────────
# Ported from Hermes: installs Chromium headless shell for the browser
# tool. Stored at $PLAYWRIGHT_BROWSERS_PATH so it survives the volume
# overlay at runtime.
RUN cd frontend/terminal && npx playwright install --with-deps chromium --only-shell 2>/dev/null || \
    echo "[build] Warning: Playwright install failed; browser tool may not work"

# ─── Source code ───────────────────────────────────────────────────────────
# .dockerignore excludes node_modules, .git, __pycache__, etc.
COPY --link --chmod=a+rX,go-w . .

# ─── Install NIA as editable (fast, no re-resolve) ─────────────────────────
RUN uv pip install --no-cache-dir --no-deps -e "."

# ─── Wire exec shim and install-method stamp ───────────────────────────────
# Ported from Hermes: the exec shim drops root to the nia user for
# `docker exec` invocations. The .install_method stamp is baked next
# to the code (NOT into $NIA_HOME) so it doesn't clobber host installs.
USER root
RUN mkdir -p /opt/nia/bin && \
    cp /opt/nia/docker/nia-exec-shim.sh /opt/nia/bin/niaharness && \
    chmod 0755 /opt/nia/bin/niaharness && \
    printf 'docker\n' > /opt/nia/.install_method

# ─── Bake build-time git revision ──────────────────────────────────────────
# Ported from Hermes: writes the commit SHA to a file at build time so
# `niaharness --version` and the startup banner can show it.
ARG NIA_GIT_SHA=
RUN if [ -n "${NIA_GIT_SHA}" ]; then \
        printf '%s\n' "${NIA_GIT_SHA}" > /opt/nia/.nia_build_sha; \
    fi

# ─── s6-overlay service wiring ─────────────────────────────────────────────
# Static services declared at build time: main-nia (placeholder) + dashboard.
# Per-profile gateway services are registered dynamically at runtime.
COPY docker/s6-rc.d/ /etc/s6-overlay/s6-rc.d/

# cont-init.d scripts: run as root before user services start.
# 01-nia-setup: stage2 hook (UID remap, volume chown, config seed, skills sync)
# 015-supervise-perms: chown s6 supervise/ trees to nia user
# 02-reconcile-profiles: recreate per-profile gateway s6 services after restart
RUN mkdir -p /etc/cont-init.d && \
    printf '#!/command/with-contenv sh\nexec /opt/nia/docker/stage2-hook.sh\n' \
        > /etc/cont-init.d/01-nia-setup && \
    chmod +x /etc/cont-init.d/01-nia-setup
COPY --chmod=0755 docker/cont-init.d/015-supervise-perms /etc/cont-init.d/015-supervise-perms
COPY --chmod=0755 docker/cont-init.d/02-reconcile-profiles /etc/cont-init.d/02-reconcile-profiles

# ─── Runtime environment ───────────────────────────────────────────────────
# Ported from Hermes: sealed /opt/nia (root-owned, read-only) prevents
# runtime self-modification. Lazy installs redirected to writable volume.
ENV NIA_HOME=/opt/data
ENV NIA_WRITE_SAFE_ROOT=/opt/data
ENV NIA_DISABLE_LAZY_INSTALLS=0
ENV NIA_LAZY_INSTALL_TARGET=/opt/data/lazy-packages
ENV NIA_TUI_DIR=/opt/nia/frontend/terminal
ENV PATH="/opt/nia/bin:/opt/nia/.venv/bin:/opt/data/.local/bin:${PATH}"

RUN mkdir -p /opt/data
VOLUME [ "/opt/data" ]

# s6-overlay's /init is PID 1. It sets up the supervision tree, runs
# /etc/cont-init.d/* (our stage2 hook + perms + reconciler), starts
# s6-rc services, then exec's its remaining argv as the container's
# "main program" with stdin/stdout/stderr inherited.
#
# ENTRYPOINT+CMD split so the wrapper is prepended to user-supplied args:
#   docker run <image>                  → /init main-wrapper.sh   (CMD default)
#   docker run <image> --print "hi"     → /init main-wrapper.sh --print hi
#   docker run <image> sleep infinity   → /init main-wrapper.sh sleep infinity
ENTRYPOINT [ "/init", "/opt/nia/docker/main-wrapper.sh" ]
CMD [ ]
