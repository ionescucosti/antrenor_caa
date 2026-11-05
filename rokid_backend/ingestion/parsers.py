"""Document parsers. Each returns a list of text blocks with page/section
metadata. Parsing is defensive: documents are untrusted input."""
from __future__ import annotations

import csv
import io
import json
import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from rokid_backend.core.text import clean_whitespace

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown", ".csv", ".json", ".xlsx"}
MAX_CELLS = 200_000
MAX_JSON_BYTES = 20 * 1024 * 1024


class ParseError(ValueError):
    pass


@dataclass
class Block:
    text: str
    page: int | None = None
    section: str | None = None


@dataclass
class ParsedDocument:
    blocks: list[Block]
    pages: int = 0
    warnings: list[str] = field(default_factory=list)
    used_ocr: bool = False

    @property
    def text_length(self) -> int:
        return sum(len(b.text) for b in self.blocks)


def sniff_type(filename: str, data: bytes) -> str:
    """Verify that the file content matches its extension. Returns the extension."""
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ParseError(f"Tip de fisier neacceptat: {ext or 'fara extensie'}.")
    head = data[:1024]
    if ext == ".pdf" and b"%PDF-" not in head:
        raise ParseError("Fisierul nu este un PDF valid.")
    if ext in (".docx", ".xlsx"):
        if not data.startswith(b"PK"):
            raise ParseError("Fisierul nu este un document Office valid.")
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                names = zf.namelist()
                needed = "word/document.xml" if ext == ".docx" else "xl/workbook.xml"
                if needed not in names:
                    raise ParseError("Arhiva Office nu contine structura asteptata.")
                if sum(info.file_size for info in zf.infolist()) > 400 * 1024 * 1024:
                    raise ParseError("Arhiva Office este prea mare dupa decompresie.")
        except zipfile.BadZipFile as exc:
            raise ParseError("Arhiva Office este corupta.") from exc
    if ext in (".txt", ".md", ".markdown", ".csv", ".json"):
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            try:
                data.decode("cp1250")
            except UnicodeDecodeError as exc:
                raise ParseError("Fisierul text nu este codificat UTF-8.") from exc
    return ext


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1250", errors="replace")


# --- PDF -------------------------------------------------------------------

def parse_pdf(data: bytes, ocr_enabled: bool, ocr_language: str) -> ParsedDocument:
    import fitz  # PyMuPDF

    blocks: list[Block] = []
    warnings: list[str] = []
    used_ocr = False
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise ParseError(f"PDF-ul nu poate fi deschis: {exc}") from exc
    with doc:
        if doc.is_encrypted and not doc.authenticate(""):
            raise ParseError("PDF-ul este protejat prin parola.")
        if len(doc) > 2000:
            raise ParseError("PDF prea mare: limita este de 2000 de pagini.")
        ocr_failed_once = False
        for index, page in enumerate(doc):
            text = clean_whitespace(page.get_text("text", sort=True))
            if len(text) < 20 and ocr_enabled and not ocr_failed_once:
                try:
                    textpage = page.get_textpage_ocr(language=ocr_language, dpi=200, full=True)
                    text = clean_whitespace(page.get_text("text", textpage=textpage, sort=True))
                    used_ocr = True
                except Exception as exc:  # Tesseract missing or failing
                    ocr_failed_once = True
                    warnings.append(f"OCR indisponibil (pagina {index + 1}): {str(exc)[:160]}")
            elif len(text) < 20 and not ocr_enabled:
                warnings.append(f"Pagina {index + 1} nu contine text extractibil (OCR dezactivat).")
            if text:
                blocks.append(Block(text=text, page=index + 1))
        pages = len(doc)
    return ParsedDocument(blocks=blocks, pages=pages, warnings=warnings, used_ocr=used_ocr)


# --- DOCX ------------------------------------------------------------------

def parse_docx(data: bytes) -> ParsedDocument:
    import docx

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise ParseError(f"DOCX-ul nu poate fi citit: {exc}") from exc
    blocks: list[Block] = []
    section: str | None = None
    buffer: list[str] = []

    def flush():
        if buffer:
            blocks.append(Block(text="\n".join(buffer), section=section))
            buffer.clear()

    for paragraph in document.paragraphs:
        text = clean_whitespace(paragraph.text)
        if not text:
            continue
        style = (paragraph.style.name if paragraph.style is not None else "") or ""
        if style.lower().startswith(("heading", "title")):
            flush()
            section = text[:200]
            continue
        buffer.append(text)
    flush()
    for table in document.tables:
        rows = []
        for row in table.rows:
            cells = [clean_whitespace(c.text) for c in row.cells]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            blocks.append(Block(text="\n".join(rows), section=section))
    return ParsedDocument(blocks=blocks, pages=0)


# --- Plain text / Markdown ---------------------------------------------------

def parse_text(data: bytes, markdown: bool) -> ParsedDocument:
    text = _decode(data).replace("\r\n", "\n")
    blocks: list[Block] = []
    section: str | None = None
    buffer: list[str] = []

    def flush():
        joined = "\n".join(buffer).strip()
        if joined:
            blocks.append(Block(text=joined, section=section))
        buffer.clear()

    for line in text.split("\n"):
        if markdown and line.lstrip().startswith("#"):
            flush()
            section = line.lstrip("# ").strip()[:200] or section
            continue
        buffer.append(line.rstrip())
    flush()
    return ParsedDocument(blocks=blocks, pages=0)


# --- CSV / XLSX ---------------------------------------------------------------

def _rows_to_blocks(header: list[str], rows: list[list[str]], section: str | None, rows_per_block: int = 20) -> list[Block]:
    blocks = []
    for start in range(0, len(rows), rows_per_block):
        lines = []
        for row in rows[start:start + rows_per_block]:
            pairs = [f"{h}: {v}" for h, v in zip(header, row) if clean_whitespace(str(v))]
            if pairs:
                lines.append("; ".join(pairs))
        if lines:
            blocks.append(Block(text="\n".join(lines), section=section))
    return blocks


def parse_csv(data: bytes) -> ParsedDocument:
    text = _decode(data)
    first_line = text.split("\n", 1)[0]
    delimiter = max(",;\t|", key=first_line.count) if any(d in first_line for d in ",;\t|") else ","

    class dialect(csv.excel):  # noqa: N801 - csv dialect class
        pass
    dialect.delimiter = delimiter
    reader = csv.reader(io.StringIO(text), dialect)
    rows = [[clean_whitespace(c) for c in row] for row in reader if any(c.strip() for c in row)]
    if not rows:
        raise ParseError("CSV-ul este gol.")
    if len(rows) * len(rows[0]) > MAX_CELLS:
        raise ParseError("CSV prea mare.")
    header = rows[0] if len(rows) > 1 else [f"col{i + 1}" for i in range(len(rows[0]))]
    body = rows[1:] if len(rows) > 1 else rows
    return ParsedDocument(blocks=_rows_to_blocks(header, body, None), pages=0)


def parse_xlsx(data: bytes) -> ParsedDocument:
    import openpyxl

    try:
        workbook = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except Exception as exc:
        raise ParseError(f"XLSX-ul nu poate fi citit: {exc}") from exc
    blocks: list[Block] = []
    cells = 0
    for sheet in workbook.worksheets:
        rows: list[list[str]] = []
        for row in sheet.iter_rows(values_only=True):
            values = ["" if v is None else clean_whitespace(str(v)) for v in row]
            if any(values):
                rows.append(values)
                cells += len(values)
                if cells > MAX_CELLS:
                    raise ParseError("XLSX prea mare.")
        if not rows:
            continue
        header = rows[0] if len(rows) > 1 else [f"col{i + 1}" for i in range(len(rows[0]))]
        blocks.extend(_rows_to_blocks(header, rows[1:] if len(rows) > 1 else rows, sheet.title))
    workbook.close()
    return ParsedDocument(blocks=blocks, pages=0)


# --- JSON ---------------------------------------------------------------------

def _flatten(value, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        out = []
        for key, item in value.items():
            out.extend(_flatten(item, f"{prefix}{key}." if not isinstance(item, (str, int, float, bool)) else f"{prefix}{key}"))
        return out
    if isinstance(value, list):
        out = []
        for index, item in enumerate(value):
            out.extend(_flatten(item, f"{prefix}[{index}]" if not isinstance(item, (dict, list)) else prefix))
        return out
    return [f"{prefix.rstrip('.')}: {clean_whitespace(str(value))}"] if prefix else [clean_whitespace(str(value))]


def parse_json(data: bytes) -> ParsedDocument:
    if len(data) > MAX_JSON_BYTES:
        raise ParseError("JSON prea mare.")
    try:
        payload = json.loads(_decode(data))
    except json.JSONDecodeError as exc:
        raise ParseError(f"JSON invalid: {exc.msg} (linia {exc.lineno}).") from exc
    blocks: list[Block] = []
    items = payload if isinstance(payload, list) else [payload]
    for index, item in enumerate(items):
        lines = _flatten(item)
        if lines:
            blocks.append(Block(text="\n".join(lines[:400]), section=f"item {index + 1}" if isinstance(payload, list) else None))
    return ParsedDocument(blocks=blocks, pages=0)


def parse_document(filename: str, data: bytes, *, ocr_enabled: bool = True, ocr_language: str = "eng") -> ParsedDocument:
    ext = sniff_type(filename, data)
    if ext == ".pdf":
        parsed = parse_pdf(data, ocr_enabled, ocr_language)
    elif ext == ".docx":
        parsed = parse_docx(data)
    elif ext in (".txt",):
        parsed = parse_text(data, markdown=False)
    elif ext in (".md", ".markdown"):
        parsed = parse_text(data, markdown=True)
    elif ext == ".csv":
        parsed = parse_csv(data)
    elif ext == ".xlsx":
        parsed = parse_xlsx(data)
    else:
        parsed = parse_json(data)
    if parsed.text_length < 20:
        hint = " OCR indisponibil sau esuat." if any("OCR" in w for w in parsed.warnings) else ""
        raise ParseError("Documentul nu contine text extractibil." + hint)
    return parsed
