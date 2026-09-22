#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PORT=8001

# ── parse args ────────────────────────────────────────────────────────

while getopts "p:h" opt; do
  case "$opt" in
    p) PORT="$OPTARG" ;;
    h)
      echo "Usage: ./start.sh [-p PORT]"
      echo "  Default port: 8001"
      echo "  Example: ./start.sh -p 8080"
      exit 0
      ;;
    *) exit 1 ;;
  esac
done

# ── check port ────────────────────────────────────────────────────────

if ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
  echo "Port $PORT is already in use. Please switch to another port:"
  echo "  ./start.sh -p <another_port>"
  exit 1
fi

# ── start server ──────────────────────────────────────────────────────

echo "Starting BITFSD Generator on http://localhost:$PORT"
uv run python src/server.py --host 0.0.0.0 --port "$PORT"
