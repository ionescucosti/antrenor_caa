"""Parsing and validation of verified question/answer datasets.

Accepted inputs: JSON (list or {"records": [...]}), JSONL, CSV and the official
ANR question PDFs (through the existing Antrenor CAA importer)."""
from __future__ import annotations

import csv
import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rokid_backend.core.text import clean_whitespace, match_key, normalize_text, question_hash, strip_option_label

LABELS = "ABCDEFGHIJKL"
QA_EXTENSIONS = {".json", ".jsonl", ".csv", ".pdf"}


@dataclass
class QARecordDraft:
    question: str
    question_type: str
    options: list[dict[str, str]]
    correct_labels: list[str]
    correct_texts: list[str]
    explanation: str = ""
    source: dict[str, Any] = field(default_factory=dict)

    @property
    def normalized_question(self) -> str:
        return normalize_text(self.question)

    @property
    def question_hash(self) -> str:
        return question_hash(self.question)


@dataclass
class ParsedDataset:
    records: list[QARecordDraft]
    errors: list[str]
    format: str
    duplicates: int = 0


class DatasetError(ValueError):
    pass


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1250", errors="replace")


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [clean_whitespace(str(v)) for v in value if clean_whitespace(str(v))]
    text = clean_whitespace(str(value))
    if not text:
        return []
    if "," in text or ";" in text or "|" in text:
        return [p.strip() for p in text.replace(";", ",").replace("|", ",").split(",") if p.strip()]
    return [text]


def _options_from(raw) -> list[dict[str, str]]:
    options: list[dict[str, str]] = []
    if isinstance(raw, dict):
        raw = [{"label": k, "text": v} for k, v in raw.items()]
    if isinstance(raw, list):
        for index, item in enumerate(raw):
            if isinstance(item, dict):
                label = clean_whitespace(str(item.get("label") or item.get("id") or LABELS[index])).upper()
                text = clean_whitespace(str(item.get("text") or item.get("value") or ""))
            else:
                text = clean_whitespace(str(item))
                label = LABELS[index] if index < len(LABELS) else str(index + 1)
                stripped = strip_option_label(text)
                if stripped != text and text[:1].upper().isalpha():
                    label, text = text[:1].upper(), stripped
            options.append({"label": label, "text": text})
    return options


def record_from_mapping(item: dict[str, Any], source: dict[str, Any] | None = None) -> QARecordDraft:
    question = clean_whitespace(str(item.get("question") or item.get("text") or ""))
    qtype = clean_whitespace(str(item.get("question_type") or "")).lower() or "multiple_choice"
    options = _options_from(item.get("options"))
    if not options:
        for letter in LABELS:
            key = f"option_{letter.lower()}"
            if key in item and clean_whitespace(str(item[key] or "")):
                options.append({"label": letter, "text": clean_whitespace(str(item[key]))})
    raw_correct = item.get("correct_answer")
    if raw_correct is None:
        raw_correct = item.get("correct") if "correct" in item else item.get("answer")
    explanation = clean_whitespace(str(item.get("explanation") or ""))
    src = dict(source or {})
    if item.get("source"):
        src.update(item["source"] if isinstance(item["source"], dict) else {"ref": str(item["source"])})
    if item.get("id"):
        src.setdefault("record_id", str(item["id"]))
    if qtype in ("multiple_choice", "single_choice", "true_false") and options:
        labels = [c.upper() for c in _as_list(raw_correct)]
        by_label = {o["label"].upper(): o["text"] for o in options}
        texts = [by_label[c] for c in labels if c in by_label]
        if not texts and labels:
            # correct_answer given as option text rather than label
            by_text = {match_key(o["text"]): o for o in options}
            found = [by_text.get(match_key(c)) for c in labels]
            if all(found):
                labels = [o["label"] for o in found]  # type: ignore[index]
                texts = [o["text"] for o in found]  # type: ignore[index]
        return QARecordDraft(question, "multiple_choice", options, labels, texts, explanation, src)
    answer_texts = _as_list(raw_correct) if isinstance(raw_correct, list) else ([clean_whitespace(str(raw_correct))] if raw_correct not in (None, "") else [])
    return QARecordDraft(question, "open", options, [], answer_texts, explanation, src)


def validate_record(draft: QARecordDraft) -> list[str]:
    problems = []
    if len(draft.normalized_question) < 3:
        problems.append("intrebare lipsa sau prea scurta")
    if draft.question_type == "multiple_choice":
        if len(draft.options) < 2:
            problems.append("sunt necesare cel putin doua optiuni")
        labels = [o["label"].upper() for o in draft.options]
        if len(labels) != len(set(labels)):
            problems.append("etichete de optiuni duplicate")
        if any(not o["text"] for o in draft.options):
            problems.append("optiune fara text")
        if not draft.correct_labels:
            problems.append("lipseste raspunsul corect")
        elif any(c not in labels for c in draft.correct_labels):
            problems.append("raspunsul corect nu corespunde optiunilor")
        if len(draft.correct_texts) != len(draft.correct_labels):
            problems.append("textul optiunii corecte nu a putut fi stabilit")
    else:
        if not draft.correct_texts:
            problems.append("lipseste raspunsul corect")
    return problems


def _parse_json_like(data: bytes, jsonl: bool) -> list[dict[str, Any]]:
    text = _decode(data)
    items: list[Any] = []
    if jsonl:
        for line_no, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise DatasetError(f"JSONL invalid la linia {line_no}: {exc.msg}") from exc
    else:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"JSON invalid: {exc.msg} (linia {exc.lineno}).") from exc
        if isinstance(payload, dict):
            for key in ("records", "questions", "items", "data"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
            else:
                payload = [payload]
        if not isinstance(payload, list):
            raise DatasetError("JSON-ul trebuie sa contina o lista de inregistrari.")
        items = payload
    return [i for i in items if isinstance(i, dict)]


def _parse_csv(data: bytes) -> list[dict[str, Any]]:
    text = _decode(data)
    first_line = text.split("\n", 1)[0]
    delimiter = max(",;\t|", key=first_line.count) if any(d in first_line for d in ",;\t|") else ","

    class dialect(csv.excel):  # noqa: N801 - csv dialect class
        pass
    dialect.delimiter = delimiter
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    rows = []
    for row in reader:
        clean = {(k or "").strip().lower().replace(" ", "_"): (v or "") for k, v in row.items()}
        if clean.get("options"):
            clean["options"] = [p.strip() for p in str(clean["options"]).split("|") if p.strip()]
        rows.append(clean)
    return rows


def _anr_importer():
    try:
        import importer  # type: ignore
        return importer
    except ImportError:
        root = Path(__file__).resolve().parents[2]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import importer  # type: ignore
        return importer


def parse_anr_pdf(path: Path, category: str) -> ParsedDataset:
    """Official ANR question lists: reuses the Antrenor CAA table importer."""
    importer = _anr_importer()
    try:
        questions, report = importer.extract_bank(path, category.upper())
    except importer.ImportFailure as exc:
        raise DatasetError(f"PDF ANR: {exc}") from exc
    records, errors = [], []
    for q in questions:
        source = {"file": path.name, "page": q["page"], "number": q["number"], "section": q["section"], "category": category.upper()}
        options = [{"label": LABELS[i], "text": text or f"[ilustratie {LABELS[i]}]"} for i, text in enumerate(q["options"])]
        labels = [LABELS[i] for i in q["correct"]]
        draft = QARecordDraft(q["text"] or "[intrebare ilustrata]", "multiple_choice", options, labels,
                              [options[i]["text"] for i in q["correct"]], q.get("warning", ""), source)
        if not q["gradable"]:
            errors.append(f"p.{q['page']} nr.{q['number']}: cheie incompleta in PDF; inregistrarea a fost omisa")
            continue
        if any(q["visuals"]):
            draft.source["has_images"] = True
        records.append(draft)
    for warning in report.get("warnings", []):
        errors.append(f"avertisment importer: {warning}")
    return ParsedDataset(records, errors, "anr_pdf")


def parse_qa_dataset(filename: str, data: bytes, *, stored_path: Path | None = None, anr_category: str | None = None) -> ParsedDataset:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        if not anr_category or anr_category.upper() not in ("C", "D"):
            raise DatasetError("Pentru PDF-urile ANR indica categoria (C sau D).")
        if stored_path is None:
            raise DatasetError("PDF-ul trebuie salvat pe disc inainte de parsare.")
        parsed = parse_anr_pdf(stored_path, anr_category)
        raw_items: list[dict[str, Any]] = []
    elif ext == ".jsonl":
        raw_items, parsed = _parse_json_like(data, True), ParsedDataset([], [], "jsonl")
    elif ext == ".json":
        raw_items, parsed = _parse_json_like(data, False), ParsedDataset([], [], "json")
    elif ext == ".csv":
        raw_items, parsed = _parse_csv(data), ParsedDataset([], [], "csv")
    else:
        raise DatasetError("Format de set Q&A neacceptat (json, jsonl, csv sau pdf ANR).")
    for index, item in enumerate(raw_items, 1):
        draft = record_from_mapping(item, {"file": Path(filename).name, "line": index})
        problems = validate_record(draft)
        if problems:
            parsed.errors.append(f"inregistrarea {index}: " + "; ".join(problems))
        else:
            parsed.records.append(draft)
    seen: set[tuple[str, tuple[str, ...]]] = set()
    unique = []
    for draft in parsed.records:
        key = (draft.question_hash, tuple(sorted(match_key(t) for t in draft.correct_texts)))
        if key in seen:
            parsed.duplicates += 1
            continue
        seen.add(key)
        unique.append(draft)
    parsed.records = unique
    if not parsed.records:
        raise DatasetError("Setul nu contine nicio inregistrare valida. " + ("; ".join(parsed.errors[:3]) if parsed.errors else ""))
    return parsed
