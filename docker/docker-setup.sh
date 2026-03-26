#!/usr/bin/env bash
# =============================================================================
# docker-setup.sh — N.I.A. Sandbox Image Builder
# =============================================================================
# Builds all three static sandbox images in the correct dependency order.
# Run this ONCE to pre-build images; N.I.A. never pulls images at runtime.
#
# Usage:
#   bash docker/docker-setup.sh              # Build all (default)
#   bash docker/docker-setup.sh --no-browser # Skip browser sandbox
#   bash docker/docker-setup.sh --verify     # Build + verify images exist
#
# Images built:
#   nia-sandbox:latest          — Base OS layer
#   nia-sandbox-common:latest   — Python 3.12 + Node 20 + common tools
#   nia-sandbox-browser:latest  — Playwright + Chromium (optional)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ── Color output ──────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[NIA]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERR]${NC}  $*"; exit 1; }

# ── CLI flags ─────────────────────────────────────────────────────────────────
BUILD_BROWSER=true
VERIFY_ONLY=false

for arg in "$@"; do
  case $arg in
    --no-browser) BUILD_BROWSER=false ;;
    --verify)     VERIFY_ONLY=true ;;
    --help|-h)
      echo "Usage: bash docker/docker-setup.sh [--no-browser] [--verify]"
      exit 0
      ;;
  esac
done

# ── Pre-flight: Docker running? ───────────────────────────────────────────────
if ! docker info &>/dev/null; then
  error "Docker is not running. Please start Docker Desktop first."
fi

if [ "$VERIFY_ONLY" = true ]; then
  info "Verifying sandbox images..."
  for image in "nia-sandbox:latest" "nia-sandbox-common:latest"; do
    if docker image inspect "$image" &>/dev/null; then
      success "$image exists"
    else
      warn "$image NOT found — run docker-setup.sh to build"
    fi
  done
  if [ "$BUILD_BROWSER" = true ]; then
    if docker image inspect "nia-sandbox-browser:latest" &>/dev/null; then
      success "nia-sandbox-browser:latest exists"
    else
      warn "nia-sandbox-browser:latest NOT found"
    fi
  fi
  exit 0
fi

# ── Build ─────────────────────────────────────────────────────────────────────
info "Building N.I.A. sandbox images from: $PROJECT_ROOT"
echo ""

# Layer 1: Base sandbox OS
info "Building nia-sandbox:latest (Layer 1 — base OS) ..."
docker build \
  -t nia-sandbox:latest \
  -f "$SCRIPT_DIR/Dockerfile.sandbox" \
  "$PROJECT_ROOT"
success "nia-sandbox:latest built"

# Layer 2: Python + Node runtimes
info "Building nia-sandbox-common:latest (Layer 2 — Python + Node) ..."
docker build \
  -t nia-sandbox-common:latest \
  -f "$SCRIPT_DIR/Dockerfile.sandbox-common" \
  "$PROJECT_ROOT"
success "nia-sandbox-common:latest built"

# Layer 3: Browser (optional)
if [ "$BUILD_BROWSER" = true ]; then
  info "Building nia-sandbox-browser:latest (Layer 3 — Playwright + Chromium) ..."
  info "(This may take a few minutes for the Chromium install)"
  docker build \
    -t nia-sandbox-browser:latest \
    -f "$SCRIPT_DIR/Dockerfile.sandbox-browser" \
    "$PROJECT_ROOT"
  success "nia-sandbox-browser:latest built"
else
  warn "Skipping browser sandbox (--no-browser flag)"
fi

# ── Final Summary ─────────────────────────────────────────────────────────────
echo ""
info "All sandbox images built successfully!"
echo ""
info "Available images:"
docker images --filter "reference=nia-sandbox*" --format "  {{.Repository}}:{{.Tag}}  ({{.Size}})"
echo ""
info "Run NIA: python main.py"
