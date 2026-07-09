#!/bin/sh
# shellcheck shell=sh
# /opt/nia/docker/stage2-hook.sh — s6-overlay stage2 hook.
#
# Ported from Hermes Agent's docker/stage2-hook.sh (542 lines).
# Runs as root after the supervision tree is up but before user
# services start. Handles:
#   1. Reject unsupported --user start
#   2. UID/GID remap (NIA_UID/NIA_GID or PUID/PGID aliases)
#   3. Docker socket group membership (for DooD terminal backend)
#   4. Data volume ownership fix (targeted, not blanket chown -R)
#   5. Seed directory structure
#   6. Config file seeding (.env, SOUL.md)
#   7. Skills sync
#   8. Playwright browser binary discovery
#
# Wired into the image as /etc/cont-init.d/01-nia-setup by the Dockerfile.

set -eu

NIA_HOME="${NIA_HOME:-/opt/data}"
INSTALL_DIR="/opt/nia"

# Drop to nia via s6-setuidgid, but skip it when already non-root.
as_nia() { [ "$(id -u)" = 0 ] || { "$@"; return; }; s6-setuidgid nia "$@"; }

# ─── 1. Reject unsupported `docker run --user <uid>:<gid>` start ───────────
# Under s6-overlay, the bootstrap (UID remap, data-volume ownership, config
# seeding) requires root, and it is skipped when the container starts non-root.
# The baked install tree under /opt/nia is intentionally root-owned and
# non-writable; mutable runtime state must live under $NIA_HOME.
cur_uid="$(id -u)"
if [ "$cur_uid" != 0 ] && [ "$cur_uid" != "$(id -u nia)" ]; then
    cat >&2 <<EOF
[nia] ERROR: container started with --user $cur_uid (an arbitrary, non-nia UID) — not supported.

To make container-written files match your HOST user, don't use --user.
Start as root (the default) and pass your host UID/GID instead:

    docker run -e NIA_UID=\$(id -u) -e NIA_GID=\$(id -g) ...

NAS users (Synology / unRAID / UGOS) can use the PUID/PGID aliases:

    docker run -e PUID=\$(id -u) -e PGID=\$(id -g) ...

The image remaps the nia user to that UID/GID at boot and chowns the data
volume, so files land owned by your host user — the same outcome --user gave,
without breaking the s6 supervision tree.
EOF
    exit 1
fi

# ─── 2. UID/GID remap ─────────────────────────────────────────────────────
# Accept PUID/PGID as aliases for NIA_UID/NIA_GID. NAS users (UGOS, Synology,
# unRAID) expect the LinuxServer.io PUID/PGID convention.
NIA_UID="${NIA_UID:-${PUID:-}}"
NIA_GID="${NIA_GID:-${PGID:-}}"

validate_uid_gid() {
    case "$1" in
        ''|*[!0-9]*) return 1 ;;
        *) [ "$1" -ge 1 ] && [ "$1" -le 65534 ] ;;
    esac
}

if [ -n "${NIA_UID:-}" ] && validate_uid_gid "$NIA_UID" && [ "$NIA_UID" != "$(id -u nia)" ]; then
    echo "[stage2] Changing nia UID to $NIA_UID"
    usermod -u "$NIA_UID" nia
fi
if [ -n "${NIA_GID:-}" ] && validate_uid_gid "$NIA_GID" && [ "$NIA_GID" != "$(id -g nia)" ]; then
    echo "[stage2] Changing nia GID to $NIA_GID"
    groupmod -o -g "$NIA_GID" nia 2>/dev/null || true
fi

# ─── 3. Docker socket group membership (DooD terminal backend) ────────────
# When the user bind-mounts the host Docker daemon socket, the socket is
# owned by the host's `docker` group. The supervised nia user is not a
# member of any group that matches the socket's GID, so every `docker`
# invocation EACCES'es.
#
# Fix: detect the socket's GID at boot and ensure /etc/group has a matching
# entry that includes nia. Idempotent across container restarts.
for sock in /var/run/docker.sock /run/docker.sock; do
    [ -S "$sock" ] || continue
    sock_gid=$(stat -c '%g' "$sock" 2>/dev/null) || continue
    [ -n "$sock_gid" ] || continue
    if id -G nia 2>/dev/null | tr ' ' '\n' | grep -qx "$sock_gid"; then
        break
    fi
    sock_group=$(getent group "$sock_gid" 2>/dev/null | cut -d: -f1)
    if [ -z "$sock_group" ]; then
        sock_group="hostdocker"
        if ! groupadd -g "$sock_gid" "$sock_group" 2>/dev/null; then
            break
        fi
    fi
    usermod -aG "$sock_group" nia 2>/dev/null || true
    break
done

# ─── 4. Fix ownership of data volume ───────────────────────────────────────
# Ported from Hermes: targeted chown — only NIA-owned subdirs, not a
# blanket chown -R that would destroy host ownership of unrelated files
# in a bind-mounted $NIA_HOME.
mkdir -p "$NIA_HOME"

actual_nia_uid=$(id -u nia)

# Symlink-safe chown helper
refuse_symlinked_path() {
    path="$1"
    while [ -n "$path" ] && [ "$path" != "/" ]; do
        if [ -L "$path" ]; then
            return 0
        fi
        if [ "$path" = "$NIA_HOME" ]; then
            break
        fi
        parent="$(dirname "$path")"
        [ "$parent" = "$path" ] && break
        path="$parent"
    done
    return 1
}

chown_nia_tree() {
    target="$1"
    if refuse_symlinked_path "$target"; then
        return 0
    fi
    chown -R nia:nia "$target" 2>/dev/null || \
        echo "[stage2] Warning: chown $target failed (rootless container?) — continuing"
}

if [ "$(stat -c %u "$NIA_HOME" 2>/dev/null)" != "$actual_nia_uid" ]; then
    echo "[stage2] Fixing ownership of $NIA_HOME (targeted) to nia ($actual_nia_uid)"
    chown nia:nia "$NIA_HOME" 2>/dev/null || true
    # NIA-owned subdirs: recursive chown is safe here
    for sub in \
        backups cron sessions logs hooks memories skills skins \
        profiles credentials delegation platforms/pairing lazy-packages; do
        if [ -e "$NIA_HOME/$sub" ]; then
            chown_nia_tree "$NIA_HOME/$sub"
        fi
    done
fi

# Always reset ownership of profiles/ and cron/ on every boot
# (docker exec defaults to root and can leave root-owned files).
for sub in profiles cron platforms/pairing pairing; do
    if [ -d "$NIA_HOME/$sub" ]; then
        chown_nia_tree "$NIA_HOME/$sub"
    fi
done

# Reset ownership of top-level state files
for f in \
    SOUL.md memory.json settings.json auth.json \
    anthropic-oauth.json \
    sessions.db sessions.db-shm sessions.db-wal \
    active_profile gateway.pid gateway.lock; do
    if [ -e "$NIA_HOME/$f" ]; then
        if ! refuse_symlinked_path "$NIA_HOME/$f"; then
            chown nia:nia "$NIA_HOME/$f" 2>/dev/null || true
        fi
    fi
done

# config.yaml permissions
if [ -f "$NIA_HOME/config.yaml" ]; then
    if ! refuse_symlinked_path "$NIA_HOME/config.yaml"; then
        chown nia:nia "$NIA_HOME/config.yaml" 2>/dev/null || true
        chmod 640 "$NIA_HOME/config.yaml" 2>/dev/null || true
    fi
fi

# ─── 5. Seed directory structure ───────────────────────────────────────────
as_nia mkdir -p \
    "$NIA_HOME/backups" \
    "$NIA_HOME/cron" \
    "$NIA_HOME/sessions" \
    "$NIA_HOME/logs" \
    "$NIA_HOME/logs/gateways" \
    "$NIA_HOME/hooks" \
    "$NIA_HOME/memories" \
    "$NIA_HOME/skills" \
    "$NIA_HOME/profiles" \
    "$NIA_HOME/credentials" \
    "$NIA_HOME/delegation" \
    "$NIA_HOME/platforms/pairing" \
    "$NIA_HOME/lazy-packages"

# ─── Install-method stamp cleanup ─────────────────────────────────────────
# Remove stale 'docker' stamp from $NIA_HOME if one is present (the real
# stamp is at /opt/nia/.install_method, baked at build time).
if [ -f "$NIA_HOME/.install_method" ]; then
    stamped="$(tr -d '[:space:]' < "$NIA_HOME/.install_method" 2>/dev/null || true)"
    if [ "$stamped" = "docker" ]; then
        rm -f "$NIA_HOME/.install_method" 2>/dev/null || true
    fi
fi

# ─── 6. Seed config files (first boot only) ───────────────────────────────
seed_one() {
    dest=$1
    src=$2
    if [ ! -f "$NIA_HOME/$dest" ] && [ -f "$INSTALL_DIR/$src" ]; then
        if ! refuse_symlinked_path "$NIA_HOME/$dest"; then
            as_nia cp "$INSTALL_DIR/$src" "$NIA_HOME/$dest"
        fi
    fi
}
seed_one ".env" ".env.example"
seed_one "SOUL.md" "docker/SOUL.md"

# .env holds API keys — restrict to owner-only access.
if [ -f "$NIA_HOME/.env" ]; then
    if ! refuse_symlinked_path "$NIA_HOME/.env"; then
        chown nia:nia "$NIA_HOME/.env" 2>/dev/null || true
        chmod 600 "$NIA_HOME/.env" 2>/dev/null || true
    fi
fi

# ─── 7. Skills sync ────────────────────────────────────────────────────────
# Sync bundled skills from the install tree to the data volume.
if [ -d "$INSTALL_DIR/src/niaharness/skills/bundled" ]; then
    as_nia "$INSTALL_DIR/.venv/bin/python" -c "
from niaharness.tools.skills_sync import sync_skills
sync_skills()
" 2>/dev/null || echo "[stage2] Warning: skills sync failed; continuing"
fi

# ─── 8. Playwright browser binary discovery ────────────────────────────────
# Ported from Hermes: locate the Chromium binary baked at build time by
# `npx playwright install chromium` and export it so the browser tool
# can find it.
if [ -z "${AGENT_BROWSER_EXECUTABLE_PATH:-}" ] && \
        [ -n "${PLAYWRIGHT_BROWSERS_PATH:-}" ] && \
        [ -d "$PLAYWRIGHT_BROWSERS_PATH" ]; then
    browser_bin=$(find "$PLAYWRIGHT_BROWSERS_PATH" -type f -executable \
        \( -name 'chrome' -o -name 'chromium' \
           -o -name 'chrome-headless-shell' -o -name 'headless_shell' \
           -o -name 'chromium-browser' \) \
        2>/dev/null | head -n 1)
    if [ -n "$browser_bin" ]; then
        echo "[stage2] Found browser binary: $browser_bin"
        mkdir -p /run/s6/container_environment
        printf '%s' "$browser_bin" > /run/s6/container_environment/AGENT_BROWSER_EXECUTABLE_PATH
    else
        echo "[stage2] Warning: no Chromium binary under $PLAYWRIGHT_BROWSERS_PATH; browser tool may fail"
    fi
fi

echo "[stage2] Setup complete; starting user services"
