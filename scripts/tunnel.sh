#!/bin/sh
# Expune serverul local pe internet prin Cloudflare, cu HTTPS.
#   sh scripts/tunnel.sh                 -> tunel rapid de test (URL aleator *.trycloudflare.com, fara cont)
#   sh scripts/tunnel.sh <TUNNEL_TOKEN>  -> tunelul tau cu nume (Cloudflare Zero Trust), domeniul configurat acolo
# Serverul trebuie pornit inainte cu: python3 porneste.py --no-browser --behind-proxy
set -eu
PORT=${ROKID_PORT:-8765}
if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared lipseste. macOS: brew install cloudflared | Ubuntu/Debian: vezi https://pkg.cloudflare.com/" >&2
  exit 1
fi
if ! curl -fsS "http://127.0.0.1:$PORT/api/v1/health" >/dev/null 2>&1; then
  echo "Serverul nu raspunde pe http://127.0.0.1:$PORT. Porneste-l intai: python3 porneste.py --no-browser --behind-proxy" >&2
  exit 1
fi
if [ "${1:-}" != "" ]; then
  exec cloudflared tunnel --no-autoupdate run --token "$1"
fi
echo "Tunel rapid de test; URL-ul https://....trycloudflare.com apare mai jos in cateva secunde. Ctrl+C il inchide." >&2
exec cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:$PORT"
