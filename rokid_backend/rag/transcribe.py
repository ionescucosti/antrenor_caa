"""Extract the question block from a photo: vision LLM first, Tesseract OCR
(through PyMuPDF) as a fallback when available."""
from __future__ import annotations

import logging

from rokid_backend.llm.base import LLMError
from rokid_backend.rag.prompts import TRANSCRIBE_PROMPT

log = logging.getLogger(__name__)
IMAGE_TYPES = {
    b"\x89PNG\r\n\x1a\n": "image/png", b"\xff\xd8\xff": "image/jpeg", b"GIF87a": "image/gif", b"GIF89a": "image/gif",
}
MAX_IMAGE_BYTES = 12 * 1024 * 1024


class TranscriptionError(ValueError):
    pass


def sniff_image(data: bytes) -> str:
    if not data:
        raise TranscriptionError("Imaginea este goala.")
    if len(data) > MAX_IMAGE_BYTES:
        raise TranscriptionError("Imaginea depaseste 12 MB.")
    for magic, mime in IMAGE_TYPES.items():
        if data.startswith(magic):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    raise TranscriptionError("Fisierul nu este o imagine PNG, JPEG, GIF sau WEBP.")


def _tesseract(data: bytes, language: str) -> str:
    import fitz
    with fitz.open(stream=data, filetype="png" if data.startswith(b"\x89PNG") else "jpg") as doc:
        page = doc[0]
        textpage = page.get_textpage_ocr(language=language, dpi=200, full=True)
        return page.get_text("text", textpage=textpage, sort=True).strip()


def transcribe_image(data: bytes, llm, settings) -> tuple[str, str]:
    """Returns (text, method). Raises TranscriptionError when nothing works."""
    mime = sniff_image(data)
    errors = []
    if hasattr(llm, "transcribe"):
        try:
            text = llm.transcribe(data, mime, TRANSCRIBE_PROMPT)
            if text.strip() == "NO_QUESTION":
                raise TranscriptionError("Nu am gasit o intrebare in imagine.")
            if text.strip():
                return text.strip(), f"llm:{llm.name}"
            errors.append("modelul a returnat text gol")
        except LLMError as exc:
            errors.append(str(exc))
    if settings.ocr_enabled:
        try:
            text = _tesseract(data, settings.ocr_language)
            if text:
                return text, "tesseract"
            errors.append("OCR nu a gasit text")
        except Exception as exc:  # noqa: BLE001 - Tesseract missing or failing
            errors.append(f"OCR indisponibil: {str(exc).splitlines()[0][:120]}")
    raise TranscriptionError("Nu am putut extrage textul din imagine. " + "; ".join(errors))
