#!/bin/sh
# shellcheck shell=sh
# /opt/nia/bin/niaharness — `docker exec` privilege-drop shim.
#
# Ported from Hermes Agent's docker/hermes-exec-shim.sh.
#
# When an operator runs `docker exec <c> niaharness ...` the default
# UID is root (0), and any file the command writes under $NIA_HOME
# (settings.json, SOUL.md, memory.json) ends up root-owned and
# unreadable to the main process running as user `nia`.
#
# This shim drops to the `nia` user before exec'ing the real venv
# binary. When invoked as non-root, it short-circuits with no overhead.

set -e

REAL="/opt/nia/.venv/bin/niaharness"

if [ ! -x "$REAL" ]; then
    echo "nia-shim: $REAL not found or not executable" >&2
    exit 127
fi

# Already non-root? Just exec the real binary.
if [ "$(id -u)" != "0" ]; then
    exec "$REAL" "$@"
fi

# Root, with opt-out set? Honor it.
case "${NIA_DOCKER_EXEC_AS_ROOT:-}" in
    1|true|TRUE|True|yes|YES|Yes)
        exec "$REAL" "$@"
        ;;
esac

# Root, no opt-out. Drop to the nia user.
export HOME="${NIA_HOME:-/opt/data}"

# Try different privilege-drop tools
if command -v su-exec >/dev/null 2>&1; then
    exec su-exec nia "$REAL" "$@"
elif command -v gosu >/dev/null 2>&1; then
    exec gosu nia "$REAL" "$@"
else
    # Fallback: use su
    exec su nia -c "$(printf '%q ' "$REAL" "$@")"
fi
