"""Shared helpers for the Rokid backend tests (settings, uploads, questions, sample files)."""
from __future__ import annotations

import io
import json
import sys
import uuid
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rokid_backend.core.config import Settings  # noqa: E402

ADMIN_TOKEN = "test-admin-token"
DEVICE_TOKEN = "test-device-token"
EXAMPLES = REPO_ROOT / "rokid_backend" / "examples"


def make_settings(tmp_path: Path, **overrides) -> Settings:
    base = dict(environment="test", data_dir=tmp_path / "data", admin_password="secret-pass", admin_api_token=ADMIN_TOKEN,
                device_tokens=DEVICE_TOKEN, embedding_provider="hash", llm_provider="mock", log_level="WARNING",
                semantic_question_threshold=0.75, semantic_question_evidence_floor=0.3, rate_limit_per_minute=600, rate_limit_burst=200,
                caa_data_dir=tmp_path / "caa")
    base.update(overrides)
    return Settings(_env_file=None, **base)


def admin_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {ADMIN_TOKEN}"}


def device_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {DEVICE_TOKEN}"}


def upload_document(client, filename: str, data: bytes, kb: str = "default", **form):
    files = {"file": (filename, io.BytesIO(data), "application/octet-stream")}
    payload = {"knowledge_base_id": kb, **form}
    return client.post("/api/v1/documents", files=files, data=payload, headers=admin_headers())


def upload_dataset(client, filename: str, data: bytes, kb: str = "default", **form):
    files = {"file": (filename, io.BytesIO(data), "application/octet-stream")}
    payload = {"knowledge_base_id": kb, **form}
    return client.post("/api/v1/qa-datasets", files=files, data=payload, headers=admin_headers())


def ask(client, question: str, options: list[tuple[str, str]] | None = None, kb: str = "default", **extra):
    """Send the question as the raw block the glasses would send: question, then 'A. ...' lines."""
    text = extra.pop("raw_ocr_text", None) or "\n".join([question] + [f"{l}. {t}" for l, t in (options or [])])
    data = {"request_id": extra.pop("request_id", None) or f"t-{uuid.uuid4().hex[:10]}", "knowledge_base_id": kb, "text": text}
    return client.post("/api/v1/questions/answer?full=1", data=data, files=extra.pop("files", None), headers=device_headers())


def example_dataset_bytes() -> bytes:
    return (EXAMPLES / "python_basics_qa.json").read_bytes()


def example_guide_bytes() -> bytes:
    return (EXAMPLES / "python_guide.md").read_bytes()


def make_pdf_bytes(pages: list[str]) -> bytes:
    import fitz
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(40, 40, 560, 800), text, fontsize=11, fontname="helv")
    data = doc.tobytes()
    doc.close()
    return data


def make_image_pdf_bytes() -> bytes:
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.draw_rect(fitz.Rect(50, 50, 300, 300), color=(0, 0, 0), fill=(0.2, 0.2, 0.2))
    data = doc.tobytes()
    doc.close()
    return data


def make_docx_bytes() -> bytes:
    import docx
    document = docx.Document()
    document.add_heading("Chapter one", level=1)
    document.add_paragraph("Tuples are immutable sequences in Python.")
    document.add_paragraph("Lists can be modified in place.")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "type", "mutable"
    table.cell(1, 0).text, table.cell(1, 1).text = "tuple", "no"
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def make_xlsx_bytes() -> bytes:
    import openpyxl
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "types"
    sheet.append(["type", "mutable", "note"])
    sheet.append(["tuple", "no", "immutable sequence"])
    sheet.append(["list", "yes", "mutable sequence"])
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def dataset_json(records: list[dict]) -> bytes:
    return json.dumps({"records": records}, ensure_ascii=False).encode("utf-8")
