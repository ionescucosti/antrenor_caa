from helpers import ADMIN_TOKEN, admin_headers, ask, device_headers


def test_health_is_public(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert r.json()["components"]["llm_provider"] == "mock"


def test_question_endpoint_requires_token(client):
    r = client.post("/api/v1/questions/answer", data={"request_id": "x", "text": "Which Python type is immutable?"})
    assert r.status_code == 401
    r = client.post("/api/v1/questions/answer", data={"request_id": "x", "text": "Which Python type is immutable?"},
                    headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


def test_device_token_cannot_administer(client):
    assert client.get("/api/v1/documents", headers=device_headers()).status_code == 403
    assert client.get("/api/v1/knowledge-bases", headers=device_headers()).status_code == 403
    assert client.get("/api/v1/admin/tokens", headers=device_headers()).status_code == 403
    assert ask(client, "Which Python type is immutable?").status_code == 200


def test_admin_token_and_session_login(client):
    assert client.get("/api/v1/documents", headers=admin_headers()).status_code == 200
    assert client.post("/api/v1/admin/login", json={"username": "admin", "password": "nope"}).status_code == 401
    r = client.post("/api/v1/admin/login", json={"username": "admin", "password": "secret-pass"})
    assert r.status_code == 200 and "rokid_admin_session" in r.cookies
    assert client.get("/api/v1/admin/me").json()["scope"] == "admin"
    # cookie-based mutation without the CSRF header is refused
    assert client.post("/api/v1/knowledge-bases", json={"id": "k1", "name": "K1"}).status_code == 403
    assert client.post("/api/v1/knowledge-bases", json={"id": "k1", "name": "K1"}, headers={"X-Requested-With": "rokid-admin"}).status_code == 201
    client.post("/api/v1/admin/logout", headers={"X-Requested-With": "rokid-admin"})
    assert client.get("/api/v1/admin/me").status_code == 401


def test_login_rate_limited(client):
    codes = [client.post("/api/v1/admin/login", json={"username": "admin", "password": "bad"}).status_code for _ in range(8)]
    assert 429 in codes


def test_device_tokens_lifecycle(client):
    created = client.post("/api/v1/admin/tokens", json={"name": "glasses", "scope": "device"}, headers=admin_headers()).json()
    assert created["token"].startswith("rk_")
    headers = {"Authorization": f"Bearer {created['token']}"}
    assert ask(client, "Which Python type is immutable?", request_id="a").status_code == 200
    r = client.post("/api/v1/questions/answer", data={"request_id": "x", "text": "Which Python type is immutable?"}, headers=headers)
    assert r.status_code == 200
    assert client.get("/api/v1/documents", headers=headers).status_code == 403
    client.delete(f"/api/v1/admin/tokens/{created['id']}", headers=admin_headers())
    r = client.post("/api/v1/questions/answer", data={"request_id": "x", "text": "Which Python type is immutable?"}, headers=headers)
    assert r.status_code == 401
    listed = client.get("/api/v1/admin/tokens", headers=admin_headers()).json()
    assert all(t.get("token") is None for t in listed)


def test_security_headers_and_request_id(client):
    r = client.get("/api/v1/health", headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123"
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert ADMIN_TOKEN not in r.text


def test_oversized_body_is_refused_early(client):
    headers = {**device_headers(), "Content-Type": "application/json", "Content-Length": str(40 * 1024 * 1024)}
    r = client.post("/api/v1/questions/answer", content=b"{}", headers=headers)
    assert r.status_code == 413 and "MB" in r.json()["detail"]
    # a 5 MB base64 photo (the Rokid full-resolution case) is within the limit
    big = {"request_id": "big-1", "question": "Which Python type is immutable?", "options": [],
           "image": {"format": "jpeg", "base64": "A" * (5 * 1024 * 1024)}}
    r = client.post("/api/v1/questions/answer", json=big, headers=device_headers())
    assert r.status_code == 200 and r.json()["request_id"] == "big-1"
