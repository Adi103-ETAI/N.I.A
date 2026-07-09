#!/bin/sh
# shellcheck shell=sh
# /opt/nia/docker/entrypoint.sh — container entrypoint.
#
# Ported from Hermes Agent's docker/main-wrapper.sh + docker/stage2-hook.sh.
#
# Runs as root (tini's child) to handle:
#   1. UID/GID remap (NIA_UID / NIA_GID or PUID / PGID aliases)
#   2. Data volume ownership fix
#   3. Directory structure seeding
#   4. Config file seeding (.env, SOUL.md)
#   5. Argument routing (bare exec vs niaharness subcommand)
#
# Then drops to the `nia` user and exec's the final command.

set -e

NIA_HOME="${NIA_HOME:-/opt/data}"
INSTALL_DIR="/opt/nia"

# ─── Drop helper ────────────────────────────────────────────────────────────
drop() { [ "$(id -u)" = 0 ] && set -- su-exec nia "$@" 2>/dev/null || set -- gosu nia "$@" 2>/dev/null || set -- runuser -u nia -- "$@"; exec "$@"; }

# Fallback: use su if no drop tool available
drop_fallback() {
    if [ "$(id -u)" = 0 ]; then
        exec su nia -c "$(printf '%q ' "$@")"
    fi
    exec "$@"
}

# ─── 1. UID/GID remap ──────────────────────────────────────────────────────
# Accept PUID/PGID as aliases (NAS convention: Synology, unRAID, UGOS).
NIA_UID="${NIA_UID:-${PUID:-}}"
NIA_GID="${NIA_GID:-${PGID:-}}"

validate_uid_gid() {
    case "$1" in
        ''|*[!0-9]*) return 1 ;;
        *) [ "$1" -ge 1 ] && [ "$1" -le 65534 ] ;;
    esac
}

if [ -n "${NIA_UID:-}" ] && validate_uid_gid "$NIA_UID" && [ "$NIA_UID" != "$(id -u nia)" ]; then
    echo "[entrypoint] Changing nia UID to $NIA_UID"
    usermod -u "$NIA_UID" nia 2>/dev/null || true
fi
if [ -n "${NIA_GID:-}" ] && validate_uid_gid "$NIA_GID" && [ "$NIA_GID" != "$(id -g nia)" ]; then
    echo "[entrypoint] Changing nia GID to $NIA_GID"
    groupmod -o -g "$NIA_GID" nia 2>/dev/null || true
fi

# ─── 2. Fix ownership of data volume ───────────────────────────────────────
mkdir -p "$NIA_HOME"

actual_nia_uid=$(id -u nia)
if [ "$(stat -c %u "$NIA_HOME" 2>/dev/null)" != "$actual_nia_uid" ]; then
    echo "[entrypoint] Fixing ownership of $NIA_HOME to nia ($actual_nia_uid)"
    chown nia:nia "$NIA_HOME" 2>/dev/null || \
        echo "[entrypoint] Warning: chown $NIA_HOME failed (rootless container?) — continuing"
fi

# Fix ownership of NIA-owned subdirs (targeted, not blanket chown -R)
for sub in sessions logs hooks memories skills profiles credentials cron delegation; do
    if [ -e "$NIA_HOME/$sub" ]; then
        chown -R nia:nia "$NIA_HOME/$sub" 2>/dev/null || true
    fi
done

# Fix ownership of top-level state files
for f in \
    SOUL.md memory.json settings.json anthropic-oauth.json \
    sessions.db sessions.db-wal sessions.db-shm \
    active_profile; do
    if [ -e "$NIA_HOME/$f" ]; then
        chown nia:nia "$NIA_HOME/$f" 2>/dev/null || true
    fi
done

# ─── 3. Seed directory structure ───────────────────────────────────────────
for dir in \
    "$NIA_HOME/sessions" \
    "$NIA_HOME/logs" \
    "$NIA_HOME/hooks" \
    "$NIA_HOME/memories" \
    "$NIA_HOME/skills" \
    "$NIA_HOME/profiles" \
    "$NIA_HOME/credentials" \
    "$NIA_HOME/cron" \
    "$NIA_HOME/delegation"; do
    mkdir -p "$dir" 2>/dev/null || true
    chown nia:nia "$dir" 2>/dev/null || true
done

# ─── 4. Seed config files (first boot only) ───────────────────────────────
# SOUL.md
if [ ! -f "$NIA_HOME/SOUL.md" ] && [ -f "$INSTALL_DIR/docker/SOUL.md" ]; then
    cp "$INSTALL_DIR/docker/SOUL.md" "$NIA_HOME/SOUL.md"
    chown nia:nia "$NIA_HOME/SOUL.md" 2>/dev/null || true
fi

# .env from .env.example
if [ ! -f "$NIA_HOME/.env" ] && [ -f "$INSTALL_DIR/.env.example" ]; then
    cp "$INSTALL_DIR/.env.example" "$NIA_HOME/.env"
    chown nia:nia "$NIA_HOME/.env" 2>/dev/null || true
    chmod 600 "$NIA_HOME/.env" 2>/dev/null || true
fi

# ─── 5. Activate venv + argument routing ───────────────────────────────────
export HOME="$NIA_HOME"
cd "$NIA_HOME"

# Activate the venv so `niaharness` is on PATH
if [ -f "$INSTALL_DIR/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    . "$INSTALL_DIR/.venv/bin/activate"
fi

echo "[entrypoint] N.I.A ready — starting..."

# Argument routing (ported from Hermes main-wrapper.sh):
#   no args              → exec `niaharness` (the default — interactive REPL)
#   first arg is exec    → exec it directly (sleep, bash, sh, …)
#   first arg is else    → exec `niaharness <args>` (subcommand passthrough)

# Drop to nia user for the main process
if [ "$(id -u)" = 0 ]; then
    # Try different privilege-drop tools
    if command -v su-exec >/dev/null 2>&1; then
        SUID_CMD="su-exec nia"
    elif command -v gosu >/dev/null 2>&1; then
        SUID_CMD="gosu nia"
    else
        # Fallback: use su
        if [ $# -eq 0 ]; then
            exec su nia -c "cd $NIA_HOME && exec niaharness"
        fi
        if command -v "$1" >/dev/null 2>&1; then
            exec su nia -c "cd $NIA_HOME && exec $(printf '%q ' "$@")"
        fi
        exec su nia -c "cd $NIA_HOME && exec niaharness $(printf '%q ' "$@")"
    fi
fi

if [ $# -eq 0 ]; then
    exec $SUID_CMD niaharness 2>/dev/null || exec niaharness
fi

if command -v "$1" >/dev/null 2>&1; then
    exec $SUID_CMD "$@" 2>/dev/null || exec "$@"
fi

exec $SUID_CMD niaharness "$@" 2>/dev/null || exec niaharness "$@"
