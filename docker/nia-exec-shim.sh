#!/bin/sh
# shellcheck shell=sh
# /opt/nia/bin/niaharness — `docker exec` privilege-drop shim.
#
# Ported from Hermes Agent's docker/hermes-exec-shim.sh (88 lines).
#
# When an operator runs `docker exec <c> niaharness ...` the default
# UID is root (0), and any file the command writes under $NIA_HOME
# ends up root-owned and unreadable to the supervised nia process.
# This shim drops to the nia user before exec'ing the real venv binary.

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

# s6-setuidgid lives under /command/ which is NOT on `docker exec`'s PATH.
S6_SUID=/command/s6-setuidgid
if [ -x "$S6_SUID" ]; then
    exec "$S6_SUID" nia "$REAL" "$@"
fi

# Fallback: try gosu, then su
if command -v gosu >/dev/null 2>&1; then
    exec gosu nia "$REAL" "$@"
fi

exec su nia -c "$(printf '%q ' "$REAL" "$@")"
