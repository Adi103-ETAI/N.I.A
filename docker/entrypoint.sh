#!/bin/sh
# shellcheck shell=sh
# Deprecated shim. The real logic lives in docker/stage2-hook.sh, invoked
# by /etc/cont-init.d/01-nia-setup (installed by the Dockerfile). This
# file exists so external references to docker/entrypoint.sh still work,
# but it's no longer the ENTRYPOINT — /init is.
#
# Ported from Hermes Agent's docker/entrypoint.sh.
echo "[nia] WARNING: docker/entrypoint.sh is a deprecated shim under " \
    "s6-overlay. The container's real ENTRYPOINT is /init + " \
    "main-wrapper.sh; this script only runs the stage2 cont-init hook " \
    "and does NOT exec the CMD." >&2
exec /opt/nia/docker/stage2-hook.sh "$@"
