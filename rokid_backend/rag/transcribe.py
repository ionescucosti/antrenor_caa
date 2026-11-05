"""Extract the question block from a photo: vision LLM first, Tesseract OCR
(through PyMuPDF) as a fallback when available. The photo is downscaled before
the model call; the model's reply is cleaned into the plain layout that
split_raw_question() expects (question line, then "A. ...", "B. ...")."""
from __future__ import annotations

import logging
import re

import time

from rokid_backend.core.config import DEFAULT_TRANSCRIBE_CONTEXT
from rokid_backend.llm.base import LLMError, extract_json
from rokid_backend.rag.imageprep import (SCREEN_SKIP_ABOVE, crop_image, exif_orientation, find_screen, image_size, prepare_image,
                                         sharpness)
from rokid_backend.rag.prompts import LOCATE_PROMPT, build_transcribe_prompt

log = logging.getLogger(__name__)
IMAGE_TYPES = {
    b"\x89PNG\r\n\x1a\n": "image/png", b"\xff\xd8\xff": "image/jpeg", b"GIF87a": "image/gif", b"GIF89a": "image/gif",
}
MAX_IMAGE_BYTES = 12 * 1024 * 1024

_FENCE_OPEN = re.compile(r"^```[A-Za-z]*[ \t]*\n?")
_FENCE_CLOSE = re.compile(r"\n?```\s*$")
_LABEL_ONLY = re.compile(r"^\s*(?:întrebare|intrebare|question|opțiuni|optiuni|options|răspunsuri|raspunsuri|answers|variante)\s*:?\s*$", re.I)
_LEAD_PREFIX = re.compile(r"^\s*(?:întrebarea?|intrebarea?|question|q)\s*(?:nr\.?\s*)?\d{0,3}\s*[:.]\s*", re.I)
_LEAD_NUMBER = re.compile(r"^\s*\(?\d{1,3}\s*[.)]\s+(?=\S)")
_NO_QUESTION = re.compile(r"^[\s\"'`*.]*no_question[\s\"'`*.]*$", re.I)


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


def clean_transcription(text: str) -> str:
    """Strip what a chatty model adds around the block: code fences, bold
    markers, 'Intrebare:' / 'Question:' labels and a leading question number."""
    text = (text or "").replace("\r\n", "\n").strip()
    text = _FENCE_CLOSE.sub("", _FENCE_OPEN.sub("", text)).replace("**", "").strip()
    if _NO_QUESTION.match(text):
        return "NO_QUESTION"
    lines = [line.rstrip() for line in text.split("\n")]
    lines = [line for line in lines if line.strip() and not _LABEL_ONLY.match(line)]
    if not lines:
        return ""
    first = _LEAD_NUMBER.sub("", _LEAD_PREFIX.sub("", lines[0], count=1), count=1).strip()
    lines[0] = first or lines[0].strip()
    return "\n".join(lines).strip()


def _tesseract(data: bytes, language: str) -> str:
    import fitz
    with fitz.open(stream=data, filetype="png" if data.startswith(b"\x89PNG") else "jpg") as doc:
        page = doc[0]
        textpage = page.get_textpage_ocr(language=language, dpi=200, full=True)
        return page.get_text("text", textpage=textpage, sort=True).strip()


def _locate_with_model(data: bytes, mime: str, llm, max_side: int = 768) -> tuple[int, int, int, int] | None:
    """Ask the vision model where the screen is, on a small thumbnail; the
    reply is a JSON box in fractions of the image size."""
    try:
        thumb, thumb_mime, _ = prepare_image(data, mime, max_side)
        reply = llm.transcribe(thumb, thumb_mime, LOCATE_PROMPT)
        box = extract_json(reply)
        if not box or box.get("found") is False:
            return None
        x0, y0, x1, y1 = (float(box[k]) for k in ("x0", "y0", "x1", "y1"))
    except (LLMError, KeyError, TypeError, ValueError) as exc:
        log.info("screen location by model failed: %s", exc)
        return None
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1 and x1 - x0 >= 0.05 and y1 - y0 >= 0.05):
        return None
    w, h = image_size(data)
    mx, my = (x1 - x0) * 0.03, (y1 - y0) * 0.05
    return (int(max(0, x0 - mx) * w), int(max(0, y0 - my) * h), int(min(1, x1 + mx) * w), int(min(1, y1 + my) * h))


def prepare_for_vision(data: bytes, mime: str, llm, settings) -> tuple[bytes, str, dict]:
    """Crop the bright screen (detector, then the model as a fallback), downscale
    if still large, refuse blurry photos. Returns (bytes, mime, info)."""
    info: dict = {}
    box = None
    if getattr(settings, "vision_crop", "auto") != "off" and not (mime == "image/jpeg" and exif_orientation(data) != 1):
        started = time.perf_counter()
        box = find_screen(data)
        info["detector_ms"] = round((time.perf_counter() - started) * 1000, 1)
        if box is None and getattr(settings, "vision_crop_model_fallback", True) and hasattr(llm, "transcribe"):
            started = time.perf_counter()
            box = _locate_with_model(data, mime, llm)
            info["locate_ms"] = round((time.perf_counter() - started) * 1000, 1)
            info["located_by_model"] = box is not None
        if box is not None:
            w, h = image_size(data)
            if (box[2] - box[0]) * (box[3] - box[1]) < SCREEN_SKIP_ABOVE * w * h:
                try:
                    data, mime = crop_image(data, box), "image/jpeg"
                    info["crop"] = box
                except Exception as exc:  # noqa: BLE001
                    log.warning("screen crop skipped: %s", type(exc).__name__)
    prepared, prepared_mime, prep = prepare_image(data, mime, getattr(settings, "vision_max_side", 1792))
    info.update(prep)
    min_sharpness = float(getattr(settings, "vision_min_sharpness", 0) or 0)
    if min_sharpness > 0:
        try:
            value = sharpness(prepared)
        except Exception as exc:  # noqa: BLE001
            log.warning("sharpness check skipped: %s", type(exc).__name__)
            value = None
        if value is not None:
            info["sharpness"] = round(value, 1)
            if value < min_sharpness:
                raise TranscriptionError(f"Poza este neclara (miscata sau nefocalizata, claritate {value:.0f} sub {min_sharpness:.0f}); reincearca tinand capul nemiscat.")
    return prepared, prepared_mime, info


def transcribe_image(data: bytes, llm, settings) -> tuple[str, str]:
    """Returns (text, method). Raises TranscriptionError when nothing works."""
    mime = sniff_image(data)
    errors = []
    if hasattr(llm, "transcribe"):
        prepared, prepared_mime, info = prepare_for_vision(data, mime, llm, settings)
        log.info("photo prepared for transcription: crop=%s located_by_model=%s sharpness=%s sent=%sx%s %s bytes (original %s) detector_ms=%s",
                 info.get("crop"), info.get("located_by_model", "-"), info.get("sharpness", "-"), info.get("width"), info.get("height"),
                 info.get("bytes", len(prepared)), info.get("original_bytes"), info.get("detector_ms"))
        prompt = build_transcribe_prompt(getattr(settings, "transcribe_context", DEFAULT_TRANSCRIBE_CONTEXT))
        try:
            text = clean_transcription(llm.transcribe(prepared, prepared_mime, prompt))
            if text == "NO_QUESTION":
                raise TranscriptionError("Nu am gasit o intrebare in imagine.")
            if text:
                return text, f"llm:{getattr(llm, 'vision_name', llm.name)}"
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
