#!/command/with-contenv sh
# shellcheck shell=sh
# /opt/nia/docker/main-wrapper.sh — wraps the container's CMD with
# argument-routing logic. Runs as /init's "main program" (Docker CMD)
# so it inherits stdin/stdout/stderr from the container.
#
# Ported from Hermes Agent's docker/main-wrapper.sh (83 lines).
#
# Routing:
#   no args                       → exec `niaharness` (the default — interactive REPL)
#   first arg is an executable    → exec it directly (sleep, bash, sh, …)
#   first arg is anything else    → exec `niaharness <args>` (subcommand passthrough)
#
# Shebang note: /init scrubs env before invoking CMD, so a plain `#!/bin/sh`
# wrapper sees an empty environ. with-contenv repopulates the env from
# /run/s6/container_environment before exec'ing.

set -e

# Drop to nia via s6-setuidgid, but skip it when already non-root.
drop() { [ "$(id -u)" = 0 ] && set -- s6-setuidgid nia "$@"; exec "$@"; }

# ─── Reject unsupported `docker run --user <uid>:<gid>` start ──────────────
cur_uid="$(id -u)"
if [ "$cur_uid" != 0 ] && [ "$cur_uid" != "$(id -u nia)" ]; then
    cat >&2 <<EOF
[nia] ERROR: container started with --user $cur_uid (an arbitrary, non-nia UID) — not supported.

To make container-written files match your HOST user, don't use --user.
Start as root (the default) and pass your host UID/GID instead:

    docker run -e NIA_UID=\$(id -u) -e NIA_GID=\$(id -g) ...

NAS users (Synology / unRAID / UGOS) can use the PUID/PGID aliases:

    docker run -e PUID=\$(id -u) -e PGID=\$(id -g) ...
EOF
    exit 1
fi

# HOME comes through with-contenv as /root. Override to the nia user's
# home so libraries that resolve paths via $HOME don't write to /root.
export HOME="${NIA_HOME:-/opt/data}"

# Save the Docker -w working directory before init scripts cd to /opt/data.
_nia_orig_cwd="${NIA_ORIG_CWD:-$PWD}"

cd "$HOME"

# Activate the venv so `niaharness` is on PATH.
# shellcheck disable=SC1091
. /opt/nia/.venv/bin/activate

# Restore the original working directory.
cd "$_nia_orig_cwd"

# ─── Argument routing ──────────────────────────────────────────────────────
if [ $# -eq 0 ]; then
    # No args → exec niaharness (interactive REPL).
    drop niaharness
fi

if command -v "$1" >/dev/null 2>&1; then
    # Bare executable — pass through directly (sleep, bash, sh, …).
    drop "$@"
fi

# NIA subcommand pass-through (--print, --list-providers, etc.).
drop niaharness "$@"
