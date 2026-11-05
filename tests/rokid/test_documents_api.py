from helpers import admin_headers, example_guide_bytes, make_docx_bytes, make_pdf_bytes, make_settings, make_xlsx_bytes, upload_document


def _doc(client, document_id):
    return client.get(f"/api/v1/documents/{document_id}", headers=admin_headers()).json()


def test_upload_and_index_supported_types(client):
    samples = {
        "guide.md": example_guide_bytes(),
        "guide.pdf": make_pdf_bytes(["Tuples are immutable. " * 10, "Lists are mutable. " * 10]),
        "guide.docx": make_docx_bytes(),
        "types.xlsx": make_xlsx_bytes(),
        "types.csv": b"type,mutable\ntuple,no\nlist,yes\n",
        "notes.txt": b"Plain text about Python tuples being immutable sequences.",
        "data.json": b'[{"type": "tuple", "mutable": false}]',
    }
    for name, data in samples.items():
        r = upload_document(client, name, data)
        assert r.status_code == 202, (name, r.text)
        doc = _doc(client, r.json()["document"]["id"])
        assert doc["status"] == "indexed", (name, doc["error"])
        assert doc["chunk_count"] >= 1
        assert doc["embedding_provider"] == "hash-ngram-512"
    pdf = next(d for d in client.get("/api/v1/documents", headers=admin_headers()).json() if d["filename"] == "guide.pdf")
    assert pdf["pages"] == 2
    assert client.app.state.store.count("chunks", "default") == sum(d["chunk_count"] for d in client.get("/api/v1/documents", headers=admin_headers()).json())


def test_duplicate_upload_is_not_reindexed(client):
    first = upload_document(client, "guide.md", example_guide_bytes()).json()
    second = upload_document(client, "copy.md", example_guide_bytes()).json()
    assert second["duplicate"] is True and second["job_id"] is None
    assert second["document"]["id"] == first["document"]["id"]
    assert len(client.get("/api/v1/documents", headers=admin_headers()).json()) == 1


def test_replace_bumps_version_and_reindexes(client):
    first = upload_document(client, "guide.md", example_guide_bytes()).json()["document"]
    replaced = upload_document(client, "guide-v2.md", b"# New\n\nCompletely new content about SQL joins and tuples.", replace_document_id=first["id"]).json()
    doc = _doc(client, first["id"])
    assert replaced["duplicate"] is False and doc["version"] == 2 and doc["filename"] == "guide-v2.md"
    assert doc["status"] == "indexed"
    assert client.app.state.store.count("chunks", "default") == doc["chunk_count"]


def test_reindex_and_delete_clean_vectors(client):
    doc = upload_document(client, "guide.md", example_guide_bytes()).json()["document"]
    before = client.app.state.store.count("chunks", "default")
    r = client.post(f"/api/v1/documents/{doc['id']}/reindex", headers=admin_headers())
    assert r.status_code == 202
    assert client.app.state.store.count("chunks", "default") == before
    job = client.get(f"/api/v1/admin/jobs/{r.json()['job_id']}", headers=admin_headers()).json()
    assert job["status"] == "done" and "parse_ms" in job["stages"]
    assert client.delete(f"/api/v1/documents/{doc['id']}", headers=admin_headers()).status_code == 204
    assert client.app.state.store.count("chunks", "default") == 0
    assert client.get(f"/api/v1/documents/{doc['id']}", headers=admin_headers()).status_code == 404


def test_rejections(client):
    assert upload_document(client, "virus.exe", b"MZ...").status_code == 400
    assert upload_document(client, "fake.pdf", b"not a pdf").status_code == 400
    assert upload_document(client, "guide.md", example_guide_bytes(), kb="missing").status_code == 400
    r = upload_document(client, "broken.json", b"{oops")
    assert r.status_code == 202
    doc = _doc(client, r.json()["document"]["id"])
    assert doc["status"] == "failed" and "JSON invalid" in doc["error"]
    status = client.get("/api/v1/admin/status", headers=admin_headers()).json()
    assert status["failed_documents"][0]["id"] == doc["id"]


def test_upload_size_limit(tmp_path, mock_llm):
    from fastapi.testclient import TestClient
    from rokid_backend.main import create_app
    app = create_app(make_settings(tmp_path, max_upload_mb=1), llm=mock_llm)
    with TestClient(app) as c:
        r = upload_document(c, "big.txt", b"x" * (1024 * 1024 + 10))
        assert r.status_code == 413


def test_knowledge_base_isolation_and_deletion(client):
    assert client.post("/api/v1/knowledge-bases", json={"id": "sql", "name": "SQL"}, headers=admin_headers()).status_code == 201
    assert client.post("/api/v1/knowledge-bases", json={"id": "sql", "name": "SQL"}, headers=admin_headers()).status_code == 409
    upload_document(client, "guide.md", example_guide_bytes(), kb="sql")
    kbs = {k["id"]: k for k in client.get("/api/v1/knowledge-bases", headers=admin_headers()).json()}
    assert kbs["sql"]["document_count"] == 1 and kbs["default"]["document_count"] == 0
    assert client.delete("/api/v1/knowledge-bases/default", headers=admin_headers()).status_code == 400
    assert client.delete("/api/v1/knowledge-bases/sql", headers=admin_headers()).status_code == 204
    assert client.get("/api/v1/documents", headers=admin_headers()).json() == []
