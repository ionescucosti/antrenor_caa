"""Antrenor CAA served by the merged server: UI, API, auth boundary."""
import re
import time

from fastapi.testclient import TestClient

from helpers import admin_headers, make_settings
from rokid_backend.llm.mock import MockLLM
from rokid_backend.main import create_app
from tests.fixtures import make_pdf as make_anr_pdf


def _client(tmp_path, loopback=True):
    app = create_app(make_settings(tmp_path), llm=MockLLM())
    return TestClient(app, client=("127.0.0.1", 50000) if loopback else ("203.0.113.9", 50000))


def test_caa_ui_and_api_work_next_to_rokid_endpoints(tmp_path):
    with _client(tmp_path) as c:
        html = c.get("/").text
        token = re.search(r'name="caa-token" content="([^"]+)"', html).group(1)
        assert token != "__TOKEN__" and c.get("/").headers["Content-Security-Policy"].startswith("default-src 'self'")
        assert c.get("/app.js").status_code == 200 and c.get("/style.css").status_code == 200
        assert c.get("/health").json()["app"] == "CAA-Trainer"
        state = c.get("/api/state").json()
        assert [x["category"] for x in state["categories"]] == ["C", "D"] and state["jobs"]["C"]["status"] == "idle"
        assert str(tmp_path / "caa") in state["dataLocation"]
        # POST without the page token is refused with the CAA error shape
        r = c.post("/api/start", json={"category": "C"})
        assert r.status_code == 403 and "error" in r.json()
        # import a synthetic ANR PDF through the CAA endpoint
        pdf = make_anr_pdf(tmp_path / "anr.pdf").read_bytes()
        r = c.post("/api/import/C", content=pdf, headers={"Content-Type": "application/pdf", "X-CAA-Token": token})
        assert r.status_code == 202, r.text
        for _ in range(200):
            job = c.get("/api/state").json()["jobs"]["C"]
            if job["status"] in ("done", "error"):
                break
            time.sleep(0.05)
        assert job["status"] == "done", job
        view = c.post("/api/start", json={"category": "C", "topic": "all", "token": token}).json()
        question = view["question"]
        assert question["number"] == 1
        answered = c.post("/api/answer", json={"sessionId": view["session"]["id"], "questionId": question["id"], "selected": [2]},
                          headers={"X-CAA-Token": token}).json()
        assert answered["feedback"]["correct"] == 1
        assert c.get(f"/crop/{question['id']}/0.png").headers["content-type"] == "image/png"
        assert c.get("/api/report/C").status_code == 200 and c.get("/source/C").headers["content-type"] == "application/pdf"
        assert c.get("/api/export?kind=runs").headers["content-type"].startswith("text/csv")
        assert c.get("/api/stats").json()["runs"][0]["answered"] == 1
        assert c.post("/api/bogus", json={"token": token}).status_code == 404
        # the bank imported in Antrenor CAA becomes a verified Q&A dataset of the backend
        r = c.post("/api/v1/qa-datasets/sync-caa", data={"knowledge_base_id": "default"}, headers=admin_headers())
        assert r.status_code == 202, r.text
        results = {x["category"]: x for x in r.json()["results"]}
        assert results["C"]["status"] == "queued" and results["D"]["status"] == "missing"
        dataset = c.get(f"/api/v1/qa-datasets/{results['C']['dataset_id']}", headers=admin_headers()).json()
        assert dataset["status"] == "indexed" and dataset["record_count"] == 3 and dataset["format"] == "anr_pdf"
        again = c.post("/api/v1/qa-datasets/sync-caa", data={"knowledge_base_id": "default"}, headers=admin_headers()).json()
        assert {x["category"]: x["status"] for x in again["results"]} == {"C": "unchanged", "D": "missing"}
        block = question["text"] + "\nA. Varianta de test C\nB. Varianta de test A\nC. Varianta de test B"
        answer = c.post("/api/v1/questions/answer?full=1", headers={"Authorization": "Bearer test-device-token"},
                        data={"request_id": "sync-1", "text": block}).json()
        assert answer["status"] == "answered" and answer["answer"]["option_labels"] == ["A"] and answer["match_type"] == "exact_qa"
        # Rokid endpoints and admin UI are served by the same process
        health = c.get("/api/v1/health").json()
        assert health["status"] == "ok" and health["components"]["antrenor_caa"] == "ok"
        assert c.get("/admin").status_code == 200


def test_remote_clients_need_admin_credentials_for_caa(tmp_path):
    with _client(tmp_path, loopback=False) as c:
        assert c.get("/").status_code == 403 and c.get("/api/state").status_code == 403
        assert c.get("/api/state", headers=admin_headers()).status_code == 200
        assert c.get("/api/v1/health").status_code == 200  # Rokid health stays public
        c.post("/api/v1/admin/login", json={"username": "admin", "password": "secret-pass"})
        assert c.get("/").status_code == 200 and c.get("/api/state").status_code == 200


def test_bad_database_url_falls_back_to_sqlite(tmp_path):
    app = create_app(make_settings(tmp_path, database_url="postgresql+psycopg://nobody:x@127.0.0.1:1/none"), llm=MockLLM())
    with TestClient(app) as c:
        health = c.get("/api/v1/health").json()
        assert health["status"] == "degraded" and health["components"]["database"] == "sqlite-fallback" and health["warnings"]
        status = c.get("/api/v1/admin/status", headers=admin_headers()).json()
        assert "SQLite" in status["database_warning"] and (tmp_path / "data" / "rokid.sqlite3").exists()


def test_tunnelled_requests_are_not_treated_as_local(tmp_path):
    """Through Cloudflare Tunnel every request reaches uvicorn from 127.0.0.1;
    the proxy headers must still keep the study UI behind the admin login."""
    with _client(tmp_path, loopback=True) as c:
        for header in ({"CF-Connecting-IP": "203.0.113.9"}, {"X-Forwarded-For": "203.0.113.9, 127.0.0.1"}):
            assert c.get("/", headers=header).status_code == 403
            assert c.get("/api/state", headers=header).status_code == 403
            assert c.get("/api/state", headers={**header, **admin_headers()}).status_code == 200
            assert c.get("/api/v1/health", headers=header).status_code == 200
        assert c.get("/").status_code == 200  # plain loopback keeps working


def test_hsts_only_behind_https(tmp_path):
    from rokid_backend.core.network import client_ip, is_local_client
    with _client(tmp_path) as c:
        assert "Strict-Transport-Security" not in c.get("/api/v1/health").headers
    app = create_app(make_settings(tmp_path / "s", cookie_secure=True), llm=MockLLM())
    with TestClient(app) as c:
        assert c.get("/api/v1/health").headers["Strict-Transport-Security"].startswith("max-age=")

    class R:  # minimal request stand-in
        def __init__(self, host, headers):
            self.client = type("C", (), {"host": host})() if host else None
            self.headers = headers
    assert client_ip(R("127.0.0.1", {"cf-connecting-ip": "198.51.100.7", "x-forwarded-for": "10.0.0.1"})) == "198.51.100.7"
    assert client_ip(R("127.0.0.1", {"x-forwarded-for": "198.51.100.8, 127.0.0.1"})) == "198.51.100.8"
    assert client_ip(R("9.9.9.9", {})) == "9.9.9.9" and not is_local_client(R("9.9.9.9", {}))
    assert is_local_client(R("::1", {})) and not is_local_client(R("::1", {"forwarded": "for=1.2.3.4"}))
