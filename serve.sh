#!/usr/bin/env bash
# Serve the website locally. The site loads templates/*.html via jQuery AJAX,
# so it must be served over HTTP (opening index.html as file:// will not work).
#
# Usage: ./serve.sh [port]   (default port 8000)
set -euo pipefail
PORT="${1:-8000}"
cd "$(dirname "$0")"
echo "Serving $(pwd) at http://localhost:${PORT}/  (Ctrl+C to stop)"
exec python3 -m http.server "${PORT}"
