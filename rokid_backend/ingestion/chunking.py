"""Chunking that keeps page/section metadata and splits on paragraph and
sentence boundaries before falling back to fixed windows."""
from __future__ import annotations

import re
from dataclasses import dataclass

from rokid_backend.ingestion.parsers import Block

_SENTENCE = re.compile(r"(?<=[.!?;:])\s+(?=[A-ZĂÂÎȘȚ0-9(\"'])")


@dataclass
class Chunk:
    text: str
    index: int
    page: int | None
    section: str | None


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    sentences = _SENTENCE.split(text)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > size:
            if current:
                pieces.append(current)
                current = ""
            step = max(1, size - overlap)
            for start in range(0, len(sentence), step):
                pieces.append(sentence[start:start + size])
                if start + size >= len(sentence):
                    break
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= size:
            current = candidate
        else:
            pieces.append(current)
            tail = current[-overlap:] if overlap and len(current) > overlap else ""
            current = f"{tail} {sentence}".strip() if tail else sentence
    if current:
        pieces.append(current)
    return [p for p in pieces if p.strip()]


def chunk_blocks(blocks: list[Block], size: int = 900, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    buffer = ""
    buffer_meta: tuple[int | None, str | None] | None = None

    def flush():
        nonlocal buffer, buffer_meta
        if buffer.strip() and buffer_meta is not None:
            for piece in _split_long(buffer.strip(), size, overlap):
                chunks.append(Chunk(text=piece, index=len(chunks), page=buffer_meta[0], section=buffer_meta[1]))
        buffer, buffer_meta = "", None

    for block in blocks:
        meta = (block.page, block.section)
        for paragraph in re.split(r"\n{2,}|\n(?=\S)", block.text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if buffer_meta is not None and buffer_meta != meta:
                flush()
            if buffer and len(buffer) + len(paragraph) + 1 > size:
                flush()
            buffer = f"{buffer}\n{paragraph}".strip() if buffer else paragraph
            buffer_meta = meta
    flush()
    return chunks
