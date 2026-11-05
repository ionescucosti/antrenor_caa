"""Split a raw OCR block ("question ... A. first B. second ...") into the
question text and labelled options. Used when a request carries no options."""
from __future__ import annotations

import re

_LABEL = re.compile(r"^\s*(?:\(?([A-Ha-h])[\.\)\]:\-]?|([1-8])[\.\)])\s+(.+?)\s*$")
_INLINE = re.compile(r"(?:(?<=\s)|^)\(?([A-H])[\.\)\]:]\s+")
LABELS = "ABCDEFGH"


def _label(match: re.Match) -> str:
    letter, digit = match.group(1), match.group(2)
    return letter.upper() if letter else LABELS[int(digit) - 1]


def split_raw_question(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (question, [(label, text), ...]). Options are recognised only as
    a sequence that starts with A (or 1) and continues in order, so a question
    line that happens to start with the word 'A' is not mistaken for an option."""
    lines = [l.rstrip() for l in (text or "").replace("\r\n", "\n").split("\n")]
    candidates = [(i, m) for i, l in enumerate(lines) if (m := _LABEL.match(l)) and l.strip()]
    for index, match in candidates:
        if _label(match) != "A":
            continue
        options: list[tuple[str, str]] = [("A", match.group(3).strip())]
        expected = "B"
        for line in lines[index + 1:]:
            if not line.strip():
                continue
            m = _LABEL.match(line)
            if m and _label(m) == expected:
                options.append((expected, m.group(3).strip()))
                expected = LABELS[LABELS.index(expected) + 1] if expected != "H" else "~"
            elif m and _label(m) in LABELS[:LABELS.index(expected)]:
                break  # labels restarting: a second question block
            else:
                label, prev = options[-1]
                options[-1] = (label, f"{prev} {line.strip()}".strip())
        if len(options) >= 2:
            question = " ".join(l.strip() for l in lines[:index] if l.strip())
            return question.strip(), options
    # single-line form: "question? A. x B. y C. z"
    flat = " ".join(l.strip() for l in lines if l.strip())
    parts = _INLINE.split(flat)
    if len(parts) >= 5:  # question, A, textA, B, textB, ...
        question, rest = parts[0].strip(), parts[1:]
        options = [(rest[i].upper(), rest[i + 1].strip()) for i in range(0, len(rest) - 1, 2)]
        if [o[0] for o in options] == list(LABELS[:len(options)]) and len(options) >= 2:
            return question, options
    return (text or "").strip(), []
