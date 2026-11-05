import pytest

from rokid_backend.ingestion.parsers import ParseError, parse_document, sniff_type
from helpers import make_docx_bytes, make_image_pdf_bytes, make_pdf_bytes, make_xlsx_bytes


def test_pdf_pages_are_preserved():
    parsed = parse_document("guide.pdf", make_pdf_bytes(["Page one talks about tuples being immutable.", "Page two talks about lists."]))
    assert parsed.pages == 2
    assert [b.page for b in parsed.blocks] == [1, 2]
    assert "tuples" in parsed.blocks[0].text


def test_docx_headings_and_tables():
    parsed = parse_document("doc.docx", make_docx_bytes())
    assert any(b.section == "Chapter one" for b in parsed.blocks)
    assert any("tuple | no" in b.text for b in parsed.blocks)


def test_markdown_sections():
    parsed = parse_document("notes.md", b"# Title\n\nSome text here.\n\n## Sub\n\nMore text here.")
    assert [b.section for b in parsed.blocks] == ["Title", "Sub"]


def test_csv_and_xlsx_rows_become_key_value_lines():
    parsed = parse_document("t.csv", b"type,mutable\ntuple,no\nlist,yes\n")
    assert "type: tuple; mutable: no" in parsed.blocks[0].text
    parsed = parse_document("t.xlsx", make_xlsx_bytes())
    assert parsed.blocks[0].section == "types"
    assert "note: immutable sequence" in parsed.blocks[0].text


def test_json_flattening():
    parsed = parse_document("d.json", b'[{"name": "tuple", "props": {"mutable": false}}]')
    assert "name: tuple" in parsed.blocks[0].text and "props.mutable: False" in parsed.blocks[0].text


def test_type_sniffing_rejects_mismatches():
    with pytest.raises(ParseError):
        sniff_type("fake.pdf", b"not a pdf at all")
    with pytest.raises(ParseError):
        sniff_type("fake.docx", b"PK\x03\x04garbage")
    with pytest.raises(ParseError):
        sniff_type("x.exe", b"MZ")
    with pytest.raises(ParseError):
        parse_document("bad.json", b"{not json")


def test_scanned_pdf_without_text():
    try:
        parsed = parse_document("scan.pdf", make_image_pdf_bytes(), ocr_enabled=True)
    except ParseError as exc:
        assert "OCR" in str(exc) or "text extractibil" in str(exc)
    else:  # Tesseract available on this machine: OCR ran (may have found little text)
        assert parsed.used_ocr
