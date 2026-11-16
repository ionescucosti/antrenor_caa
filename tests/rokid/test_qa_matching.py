import json

from rokid_backend.qa.dataset import parse_qa_dataset
from rokid_backend.qa.service import map_correct_to_options
from rokid_backend.schemas.api import Option
from helpers import admin_headers, ask, dataset_json, example_dataset_bytes, upload_dataset


def _opts(*texts):
    return [Option(label=chr(65 + i), text=t) for i, t in enumerate(texts)]


def test_option_mapping_by_text_not_letter():
    m = map_correct_to_options(["Tuple"], _opts("Dictionary", "Tuple", "List", "Set"), 0.86, 0.08)
    assert m.confident and m.labels == ["B"]
    m = map_correct_to_options(["{}", "dict()"], _opts("dict()", "[]", "{}", "set()"), 0.86, 0.08)
    assert m.confident and sorted(m.labels) == ["A", "C"]
    m = map_correct_to_options(["c. Tuple"], _opts("D. Tuple", "List"), 0.86, 0.08)
    assert m.confident and m.labels == ["A"]


def test_option_mapping_ambiguity_and_absence():
    m = map_correct_to_options(["immutable sequence tuple"], _opts("immutable sequence list", "immutable sequence tuples", "immutable sequence tupl"), 0.86, 0.08)
    assert not m.confident
    m = map_correct_to_options(["Tuple"], _opts("List", "Dictionary", "Set"), 0.86, 0.08)
    assert not m.confident
    m = map_correct_to_options(["Tuple"], [], 0.86, 0.08)
    assert m.confident and m.labels == [] and m.texts == ["Tuple"]


def test_dataset_validation_reports_each_problem():
    records = [
        {"question": "ok question?", "options": ["a", "b"], "correct_answer": "A"},
        {"question": "", "options": ["a", "b"], "correct_answer": "A"},
        {"question": "missing key?", "options": ["a", "b"]},
        {"question": "bad label?", "options": ["a", "b"], "correct_answer": "Z"},
        {"question": "one option?", "options": ["a"], "correct_answer": "A"},
        {"question": "ok question?", "options": ["a", "b"], "correct_answer": "A"},
        {"question": "open one", "question_type": "open", "correct_answer": "42"},
        {"question": "text answer?", "options": ["alpha", "beta"], "correct_answer": "beta"},
    ]
    parsed = parse_qa_dataset("set.json", dataset_json(records))
    assert len(parsed.records) == 3 and parsed.duplicates == 1
    assert len(parsed.errors) == 4
    assert parsed.records[2].correct_labels == ["B"] and parsed.records[2].correct_texts == ["beta"]


def test_csv_and_jsonl_datasets():
    csv_data = b"question,option_a,option_b,option_c,correct_answer,explanation\nWhich is immutable?,List,Tuple,Set,B,Tuples are immutable\n"
    parsed = parse_qa_dataset("set.csv", csv_data)
    assert parsed.records[0].correct_texts == ["Tuple"]
    jsonl = b"\n".join(json.dumps(r).encode() for r in [{"question": "question one?", "options": ["x", "y"], "correct_answer": ["A", "B"]}])
    parsed = parse_qa_dataset("set.jsonl", jsonl)
    assert parsed.records[0].correct_labels == ["A", "B"]


def _install_example(client):
    r = upload_dataset(client, "python_basics_qa.json", example_dataset_bytes(), name="python basics")
    assert r.status_code == 202, r.text
    ds = client.get(f"/api/v1/qa-datasets/{r.json()['dataset']['id']}", headers=admin_headers()).json()
    assert ds["status"] == "indexed" and ds["record_count"] == 5, ds
    return ds


def test_exact_match_answers_without_llm(client, mock_llm):
    ds = _install_example(client)
    r = ask(client, "Which Python type is immutable?", [("A", "List"), ("B", "Dictionary"), ("C", "Tuple"), ("D", "Set")], request_id="req-42")
    body = r.json()
    assert body["status"] == "answered" and body["request_id"] == "req-42" and body["match_type"] == "exact_qa"
    assert body["answer"] == {"option_labels": ["C"], "text": "Tuple", "explanation": "Tuples are immutable."}
    assert body["sources"][0]["document_id"] == ds["id"] and body["sources"][0]["kind"] == "qa_dataset"
    assert "llm" not in body["timings_ms"] and body["timings_ms"]["exact_match"] >= 0
    assert mock_llm.calls == []


def test_reordered_options_get_current_label(client):
    _install_example(client)
    r = ask(client, "which python type is IMMUTABLE ?", [("A", "Set"), ("B", "Tuple"), ("C", "List"), ("D", "Dictionary")])
    assert r.json()["answer"]["option_labels"] == ["B"]
    r = ask(client, "Which of the following are valid ways to create an empty dictionary?", [("A", "set()"), ("B", "dict()"), ("C", "{}"), ("D", "[]")])
    assert sorted(r.json()["answer"]["option_labels"]) == ["B", "C"]


def test_exact_match_with_unmappable_options_abstains(client, mock_llm):
    _install_example(client)
    r = ask(client, "Which Python type is immutable?", [("A", "List"), ("B", "Dictionary"), ("C", "Set"), ("D", "Frozenset")])
    body = r.json()
    assert body["status"] == "insufficient_context" and body["answer"] is None and "mapat" in body["message"]
    assert mock_llm.calls == []


def test_open_question_against_verified_record(client):
    _install_example(client)
    r = ask(client, "What does len('abc') return?")
    body = r.json()
    assert body["status"] == "answered" and body["answer"]["text"] == "3" and body["answer"]["option_labels"] == []
    r = ask(client, "Which Python type is immutable?")
    assert r.json()["answer"]["text"] == "Tuple"


def test_conflicting_verified_records_abstain(client):
    _install_example(client)
    conflicting = dataset_json([{"question": "Which Python type is immutable?", "options": ["List", "Dictionary", "Tuple", "Set"], "correct_answer": "A"}])
    r = upload_dataset(client, "conflict.json", conflicting)
    assert r.status_code == 202
    r = ask(client, "Which Python type is immutable?", [("A", "List"), ("B", "Dictionary"), ("C", "Tuple"), ("D", "Set")])
    assert r.json()["status"] == "insufficient_context" and "conflict" in r.json()["message"]


def test_same_answer_with_a_longer_wording_is_not_a_conflict(client):
    """The ANR lists repeat some questions with the correct option worded a little longer. Without options on the
    request (a photo where only the question was readable) both records are the same answer."""
    question = "Reusita ancorarii depinde si de folosirea corecta a lungimii paramei/lantului de ancorare si anume:"
    short = "lungimea paramei/lantului la ancora trebuie sa fie de 3-5 ori adancimea"
    others = ["nu sunt stabilite reguli speciale privind lungimea lantului", "lungimea paramei nu influenteaza siguranta ancorarii"]
    for name, options, correct in (("v1.json", others + [short], "C"), ("v2.json", [short + " masurata"] + others, "A")):
        r = upload_dataset(client, name, dataset_json([{"question": question, "options": options, "correct_answer": correct}]))
        assert r.status_code == 202
    body = ask(client, question).json()
    assert body["status"] == "answered" and body["match_type"] == "exact_qa", body
    assert body["answer"]["text"] == short + " masurata" and body["answer"]["option_labels"] == []
    # a different answer for the same question is still a conflict
    r = upload_dataset(client, "v3.json", dataset_json([{"question": question, "options": others + [short], "correct_answer": "A"}]))
    assert r.status_code == 202
    body = ask(client, question).json()
    assert body["status"] == "insufficient_context" and "conflict" in body["message"], body


def test_semantic_match_of_paraphrased_question(client, mock_llm):
    _install_example(client)
    r = ask(client, "Which Python type is immutable", [("A", "Set"), ("B", "Tuple"), ("C", "List"), ("D", "Dictionary")])
    body = r.json()
    assert body["status"] == "answered" and body["match_type"] in ("semantic_qa", "exact_qa") and body["answer"]["option_labels"] == ["B"]
    # paraphrased stem, reordered options: still answered from the verified record
    r = ask(client, "Which SQL clause is used to filter rows after aggregation?", [("A", "HAVING"), ("B", "WHERE"), ("C", "GROUP BY"), ("D", "ORDER BY")])
    body = r.json()
    assert body["status"] == "answered" and body["answer"]["option_labels"] == ["A"], body
    assert body["match_type"] in ("semantic_qa", "rag")
    assert body["sources"][0]["kind"] == "qa_dataset"
    # a verified record that is only loosely similar is offered to the LLM as evidence, not trusted blindly
    r = ask(client, "Which SQL clause is used to filter rows after aggregation?", [("A", "HAVING"), ("B", "WHERE"), ("C", "LIMIT")])
    body = r.json()
    assert body["match_type"] != "exact_qa"
    if body["status"] == "answered":
        assert body["answer"]["option_labels"] == ["A"] and len(mock_llm.calls) >= 1


def test_dataset_reindex_delete_and_duplicate(client):
    ds = _install_example(client)
    dup = upload_dataset(client, "again.json", example_dataset_bytes()).json()
    assert dup["duplicate"] is True
    assert client.app.state.store.count("questions", "default") == 5
    r = client.post(f"/api/v1/qa-datasets/{ds['id']}/reindex", headers=admin_headers())
    assert r.status_code == 202 and client.app.state.store.count("questions", "default") == 5
    records = client.get(f"/api/v1/qa-datasets/{ds['id']}/records?q=immutable", headers=admin_headers()).json()
    assert records["total"] == 1 and records["records"][0]["correct_texts"] == ["Tuple"]
    assert client.delete(f"/api/v1/qa-datasets/{ds['id']}", headers=admin_headers()).status_code == 204
    assert client.app.state.store.count("questions", "default") == 0
    r = ask(client, "Which Python type is immutable?", [("A", "List"), ("B", "Tuple")])
    assert r.json()["status"] == "insufficient_context"


def test_invalid_dataset_is_reported(client):
    r = upload_dataset(client, "bad.json", b'{"records": [{"question": "", "options": ["a"]}]}')
    ds = client.get(f"/api/v1/qa-datasets/{r.json()['dataset']['id']}", headers=admin_headers()).json()
    assert ds["status"] == "failed" and "nicio inregistrare valida" in ds["error"]
    r = upload_dataset(client, "x.pdf", b"%PDF-1.4 fake")
    assert r.status_code == 202
    ds = client.get(f"/api/v1/qa-datasets/{r.json()['dataset']['id']}", headers=admin_headers()).json()
    assert ds["status"] == "failed" and "categoria" in ds["error"]


def test_anr_pdf_dataset_via_existing_importer(client, tmp_path):
    from tests.fixtures import make_pdf as make_anr_pdf  # Antrenor CAA synthetic ANR table
    path = make_anr_pdf(tmp_path / "anr.pdf")
    r = upload_dataset(client, "Lista intrebari clasa C.pdf", path.read_bytes(), anr_category="C")
    assert r.status_code == 202, r.text
    ds = client.get(f"/api/v1/qa-datasets/{r.json()['dataset']['id']}", headers=admin_headers()).json()
    assert ds["status"] == "indexed" and ds["record_count"] == 3 and ds["format"] == "anr_pdf", ds
    records = client.get(f"/api/v1/qa-datasets/{ds['id']}/records", headers=admin_headers()).json()["records"]
    first = next(x for x in records if x["source"]["number"] == 1 and x["source"]["section"] == 1)
    assert first["correct_labels"] == ["C"] and first["source"]["page"] == 1
    r = ask(client, first["question"], [("A", "Varianta de test C"), ("B", "Varianta de test A"), ("C", "Varianta de test B")])
    assert r.json()["status"] == "answered" and r.json()["answer"]["option_labels"] == ["A"]
    assert r.json()["sources"][0]["page"] == 1
