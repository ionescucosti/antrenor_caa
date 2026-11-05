"""Authentication primitives: bearer tokens with scopes, signed admin sessions,
password verification and a small in-memory rate limiter."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time

SCOPE_DEVICE = "device"
SCOPE_ADMIN = "admin"


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def generate_token(prefix: str = "rk") -> str:
    return f"{prefix}_{secrets.token_urlsafe(32)}"


def hash_password(password: str, iterations: int = 200_000) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations)
    return f"pbkdf2_sha256${iterations}${salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    if not stored:
        return False
    if stored.startswith("pbkdf2_sha256$"):
        try:
            _, iterations, salt, digest = stored.split("$", 3)
            candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), int(iterations)).hex()
            return hmac.compare_digest(candidate, digest)
        except ValueError:
            return False
    return hmac.compare_digest(password.encode("utf-8"), stored.encode("utf-8"))


class SessionSigner:
    """HMAC-signed, expiring session values stored in an HttpOnly cookie."""

    def __init__(self, secret: str) -> None:
        self._key = secret.encode("utf-8")

    def sign(self, payload: dict, ttl_seconds: int) -> str:
        body = dict(payload, exp=int(time.time()) + ttl_seconds, nonce=secrets.token_hex(8))
        raw = base64.urlsafe_b64encode(json.dumps(body, separators=(",", ":")).encode()).decode().rstrip("=")
        mac = hmac.new(self._key, raw.encode(), hashlib.sha256).hexdigest()
        return f"{raw}.{mac}"

    def verify(self, value: str | None) -> dict | None:
        if not value or "." not in value:
            return None
        raw, mac = value.rsplit(".", 1)
        expected = hmac.new(self._key, raw.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(mac, expected):
            return None
        try:
            padded = raw + "=" * (-len(raw) % 4)
            body = json.loads(base64.urlsafe_b64decode(padded.encode()))
        except (ValueError, json.JSONDecodeError):
            return None
        if body.get("exp", 0) < time.time():
            return None
        return body


class RateLimiter:
    """Token bucket per client key. Memory only: adequate for a single-instance deployment."""

    def __init__(self, per_minute: int, burst: int) -> None:
        self.rate = per_minute / 60.0
        self.capacity = max(1, burst)
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, (float(self.capacity), now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            if tokens < 1:
                self._buckets[key] = (tokens, now)
                return False
            self._buckets[key] = (tokens - 1, now)
            if len(self._buckets) > 10_000:
                stale = [k for k, (_, t) in self._buckets.items() if now - t > 600]
                for k in stale:
                    self._buckets.pop(k, None)
            return True


def load_or_create_secret(path) -> str:
    """Persist a random session secret next to the data so restarts keep sessions valid."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path.read_text().strip()
    secret = secrets.token_urlsafe(48)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(secret)
    return secret
