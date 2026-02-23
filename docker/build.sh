#!/bin/bash
# =============================================================================
# N.I.A. Sandbox Image Builder
# =============================================================================
# Builds the two-layer Docker architecture in dependency order.
#
# Usage: bash docker/build.sh
# =============================================================================

set -e

echo "═══════════════════════════════════════════════"
echo "  N.I.A. Sandbox Image Builder"
echo "═══════════════════════════════════════════════"
echo ""

echo "🔨 Building Layer 1: nia-sandbox (Base OS + GUI Tools)..."
docker build -t nia-sandbox:latest -f docker/Dockerfile.sandbox .
echo "✅ Layer 1 complete."
echo ""

echo "🔨 Building Layer 2: nia-sandbox-common (Runtimes + Pi-Mono)..."
docker build -t nia-sandbox-common:latest -f docker/Dockerfile.sandbox-common .
echo "✅ Layer 2 complete."
echo ""

echo "═══════════════════════════════════════════════"
echo "  ✅ All images built successfully!"
echo ""
echo "  Images:"
echo "    nia-sandbox:latest         (Base OS)"
echo "    nia-sandbox-common:latest  (Full Agent)"
echo "═══════════════════════════════════════════════"
