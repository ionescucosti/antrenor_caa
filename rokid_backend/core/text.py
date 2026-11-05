"""Text normalisation and hashing helpers shared by ingestion and matching."""
from __future__ import annotations

import hashlib
import re
import unicodedata

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_LABEL_PREFIX = re.compile(r"^\s*(?:[A-Ha-h]|\d{1,2})\s*[\.\)\]:-]\s+")


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


def normalize_text(text: str) -> str:
    """Lower-case, accent-free, punctuation-free, single-spaced text.
    Used for exact matching of questions and answer options."""
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("ş", "ș").replace("Ş", "Ș")
    text = text.replace("ţ", "ț").replace("Ţ", "Ț")
    text = strip_accents(text.lower())
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def match_key(text: str) -> str:
    """Case/accent/whitespace-insensitive key that keeps code punctuation so
    that options such as '{}', '[]' or 'dict()' remain distinguishable."""
    text = unicodedata.normalize("NFC", text or "")
    text = strip_accents(text.lower())
    text = _WS.sub(" ", text).strip()
    return text.strip(" .;,:!?")


def strip_option_label(text: str) -> str:
    """Remove a leading 'A.' / 'b)' / '1-' label from an answer option."""
    return _LABEL_PREFIX.sub("", text or "", count=1).strip()


def clean_whitespace(text: str) -> str:
    return _WS.sub(" ", (text or "").replace("\x00", "")).strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def question_hash(question: str) -> str:
    return sha256_text(normalize_text(question))


def letters_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = sum(c.isalpha() for c in text)
    return letters / max(1, len([c for c in text if not c.isspace()]))
