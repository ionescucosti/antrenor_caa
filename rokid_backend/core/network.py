"""Client address helpers for a server that may sit behind Cloudflare Tunnel or
another reverse proxy. A request counts as *local* only when it comes from a
loopback socket AND carries no proxy headers: through cloudflared every request
reaches uvicorn from 127.0.0.1, so the socket address alone would unlock the
Antrenor CAA study UI for the whole internet."""
from __future__ import annotations

from fastapi import Request

LOOPBACK = {"127.0.0.1", "::1", "localhost"}
PROXY_HEADERS = ("cf-connecting-ip", "x-forwarded-for", "x-real-ip", "forwarded")


def socket_host(request: Request) -> str:
    return request.client.host if request.client else ""


def client_ip(request: Request) -> str:
    """Best-effort public address: Cloudflare's header first, then the first
    X-Forwarded-For hop, then the socket address (uvicorn --proxy-headers may
    already have rewritten it)."""
    cf = request.headers.get("cf-connecting-ip", "").strip()
    if cf:
        return cf
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip() or socket_host(request)
    return socket_host(request)


def is_local_client(request: Request) -> bool:
    if socket_host(request) not in LOOPBACK:
        return False
    return not any(request.headers.get(h) for h in PROXY_HEADERS)
