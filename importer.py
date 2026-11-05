"""Read numbered ANR question tables without guessing missing answer keys.

PyMuPDF is imported lazily: the UI and saved text questions do not require
network access. The original PDFs are retained for figures and auditing.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Any

TOPICS = [
    "Lumini si semnale vizuale",
    "Semnale sonore",
    "Reguli de drum si coliziuni",
    "Balizaj si navigatie interioara",
    "Navigatie si instrumente",
    "Constructie si manevre",
    "Meteorologie",
    "Siguranta si prim ajutor",
    "Legislatie si responsabilitati",
    "Notiuni generale",
]


class ImportFailure(ValueError):
    pass


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\u015f", "\u0219").replace("\u015e", "\u0218")
    text = text.replace("\u0163", "\u021b").replace("\u0162", "\u021a")
    return re.sub(r"\s+", " ", text).strip()


def plain(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text.lower())
                   if not unicodedata.combining(c))


def topic_for(question: str, choices: list[str]) -> str:
    # These are study filters, not official ANR chapter titles.
    s = plain(question + " " + " ".join(choices))
    rules = [
        (7, r"hemorag|garou|plag|victim|resuscit|respirat.*artificial|masaj.*card|prim.?ajutor|fractur|arsur|hipoterm|trusa|stingator|incend|om la apa|salvare|salvamar|epistaxis"),
        (6, r"baromet|anemomet|meteor|izobar|briza|beaufort|ciclon|anticiclon|nori|norii|presiun.*atmos|furtun|umidit|front.*rece|front.*cald"),
        (8, r"infract|contravent|sanctiun|amenda|contract|inmatric|atestare|suspendare|ultraj|pavilion.*roman|preluare.*comenzii|document.*bord"),
        (1, r"sunet|fluier|semnal.*fonic|semnal.*sonor|clopot|gong"),
        (0, r"lumin|sfera|bicon|cilindru|conuri|doua con|semn.*zi|romb"),
        (3, r"baliz|geamandur|bifurcat|ecluz|podur|pod |panou|sen(al|a)|malul|maluri|semnal.*interz|semnal.*oblig"),
        (4, r"latitud|longitud|relevment|compas|declinat|deviati|meridian|paralel|ecuator|harta|hartii|hartile|loch|sonda|radar|gps|noduri|mila marina|grad.*minut|nordul"),
        (2, r"colizi|abordaj|incrucis|depas|drum|tribord.*babord|viteza de siguranta|veghe|in vedere|vizibilitate redusa|separare.*trafic|colreg"),
        (5, r"chila|carena|etrav|etambou|pescaj|bord.*liber|stabilit|flotabil|deplasament|elice|carma|ancor|acost|amar|manevr|propulsi|motor|santina|baba|vinci|parame|punte|prova|pupa"),
    ]
    for idx, pattern in rules:
        if re.search(pattern, s):
            return TOPICS[idx]
    return TOPICS[9]


def _fitz():
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz
        except ImportError as exc:
            raise ImportFailure(
                "Lipseste PyMuPDF. Inchide aplicatia si ruleaza porneste.py "
                "pentru instalarea dependentei.") from exc
    return fitz


def _page_chars(page, fitz) -> list[tuple[Any, str]]:
    """Every character on the page with its bbox, in reading order."""
    chars = []
    for block in page.get_text("rawdict", sort=True)["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                for ch in span["chars"]:
                    chars.append((fitz.Rect(ch["bbox"]), ch["c"]))
            chars.append((None, "\n"))
    return chars


def _cell_text(chars, rect, tip_band=None) -> str:
    """Text of the characters whose centre lies inside the cell.

    PyMuPDF's clip extraction keeps any glyph that merely touches a cell, so
    a hyphen at the end of an answer line leaks into the adjacent Corect cell
    and the ANR 'Tip' value, printed on top of the rule between the question
    and the first answer, is glued to both. Digits sitting on that rule are
    skipped when tip_band is given.
    """
    out = []
    for box, ch in chars:
        if box is None:
            out.append(" ")
            continue
        cx, cy = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
        if not (rect.x0 <= cx < rect.x1 and rect.y0 <= cy < rect.y1):
            continue
        if tip_band and ch.isdigit():
            overlap = min(box.x1, tip_band[1]) - max(box.x0, tip_band[0])
            if overlap > box.width / 2:
                continue
        out.append(ch)
    return normalize("".join(out))


def _is_visual(rect, images, drawings, fitz) -> bool:
    inner = fitz.Rect(rect.x0 + 1, rect.y0 + 1, rect.x1 - 1, rect.y1 - 1)
    for info in images:
        box = fitz.Rect(info["bbox"])
        if not (inner & box).is_empty and (inner & box).get_area() > 4:
            return True
    for path in drawings:
        # Inspect individual primitives too: one PDF drawing path can contain
        # table borders and multiple symbols, with a page-sized bounding box.
        for item in path.get("items", []):
            if item[0] == "l":
                a, b = item[1:3]
                if inner.contains(a) and inner.contains(b) and abs(a - b) > 3:
                    return True
            elif item[0] == "re":
                b = fitz.Rect(item[1])
                if inner.contains(b) and b.width > 2 and b.height > 2:
                    return True
            elif item[0] == "c":
                pts = item[1:5]
                if all(inner.contains(p) for p in pts):
                    return True
            elif item[0] == "qu":
                if inner.contains(item[1].rect):
                    return True
        box = fitz.Rect(path["rect"])
        if not box.is_empty and inner.contains(box) and box.width > 2 and box.height > 2:
            return True
    return False


def extract_bank(pdf: Path, category: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fail closed on unreadable/missing question rows; never invent answers.

    A fully read question with an incomplete official key is retained but is
    ungraded. All three marking cells must explicitly contain 0/1 for grading.
    """
    if category not in ("C", "D"):
        raise ImportFailure("Categoria trebuie sa fie C sau D.")
    fitz = _fitz()
    data = pdf.read_bytes()
    if b"%PDF-" not in data[:1024]:
        raise ImportFailure("Fisierul nu este PDF (poate fi pagina de eroare 429).")
    digest = hashlib.sha256(data).hexdigest()
    report: dict[str, Any] = {
        "category": category, "sha256": digest,
        "parser": "CAA table importer 1.1", "pages": [], "warnings": [],
        "note": "Temele sunt atribuite automat. Cheia este preluata din celulele Corect.",
    }
    questions: list[dict[str, Any]] = []
    previous_number = 0
    section = 1
    found_indices: set[tuple[int, int]] = set()
    with fitz.open(pdf) as doc:
        if doc.is_encrypted:
            raise ImportFailure("PDF-ul este protejat prin parola.")
        if len(doc) > 250:
            raise ImportFailure("PDF prea mare: limita este de 250 de pagini.")
        report["pageCount"] = len(doc)
        for page_index, page in enumerate(doc):
            drawings = page.get_drawings()
            images = page.get_image_info()
            chars = _page_chars(page, fitz)
            tables = []
            for tolerance in (2, 1, 3):
                finder = page.find_tables(
                    snap_tolerance=tolerance, join_tolerance=3,
                    intersection_tolerance=3, paths=drawings)
                tables = [t for t in finder.tables
                          if t.col_count >= 7 and t.bbox[2] - t.bbox[0] > page.rect.width * .55]
                if tables:
                    break
            if not tables:
                raise ImportFailure(
                    f"Pagina {page_index + 1}: nu pot identifica tabelul ANR. "
                    "Nu a fost importata o banca partiala. Foloseste PDF-ul original, nu o scanare.")
            page_rows = []
            index_areas = []
            seen_row_boxes = set()
            for table in sorted(tables, key=lambda t: (t.bbox[1], t.bbox[0])):
                for row in table.rows:
                    cells = []
                    for cell in row.cells:
                        if cell is None:
                            continue
                        box = fitz.Rect(cell)
                        text = _cell_text(chars, box)
                        # Some ANR PDFs have a virtually zero-width 'Tip' column.
                        # A sliver this narrow cannot hold content; any text in it
                        # is bleed from a neighbour (e.g. a hyphen at a line end).
                        if box.width < 7 and not re.search(r"[0-9A-Za-z\u00C0-\u024F]", text):
                            continue
                        cells.append((box, text))
                    if not cells:
                        continue
                    first = cells[0][1]
                    number_match = re.fullmatch(r"(\d{1,5})[.)]?", first)
                    if not number_match:
                        joined = plain(" ".join(text for _, text in cells))
                        if (not joined or "raspuns" in joined or "intrebare" in joined
                                or "corect" in joined):
                            continue
                        raise ImportFailure(
                            f"Pagina {page_index + 1}: rand de tabel fara numar lizibil. "
                            "Importul a fost oprit pentru a nu pierde continut.")
                    number = int(number_match.group(1))
                    signature = tuple(round(v, 1) for v in cells[0][0])
                    if signature in seen_row_boxes:
                        continue
                    seen_row_boxes.add(signature)
                    index_areas.append(cells[0][0])
                    rest = cells[1:]
                    # A stray vertical rule past the last Corect column yields an
                    # empty trailing cell; drop it only while it holds nothing.
                    while (len(rest) > 7 and not rest[-1][1]
                           and not _is_visual(rest[-1][0], images, drawings, fitz)):
                        rest.pop()
                    if len(rest) < 7:
                        raise ImportFailure(
                            f"Pagina {page_index + 1}, intrebarea {number}: "
                            f"am identificat {len(rest)} celule, nu intrebarea si cele 6 celule de raspuns/cheie.")
                    answers_and_flags = rest[-6:]
                    question_cells = rest[:-6]
                    # The actual question is the widest leading cell; a separate
                    # narrow 'Tip' field must not be appended to the question.
                    qbox, text = max(question_cells, key=lambda item: item[0].width)
                    if any(t and b.width > 35 and b != qbox for b, t in question_cells):
                        raise ImportFailure(f"Pagina {page_index + 1}, intrebarea {number}: coloane ambigue.")
                    answer_cells = answers_and_flags[::2]
                    key_cells = answers_and_flags[1::2]
                    # Drop the 'Tip' digits printed across the rule after the question.
                    tip_band = (qbox.x1 - 4, qbox.x1 + 4)
                    text = _cell_text(chars, qbox, tip_band)
                    answer_cells[0] = (answer_cells[0][0], _cell_text(chars, answer_cells[0][0], tip_band))
                    if any(b.width > 50 for b, _ in key_cells):
                        raise ImportFailure(f"Pagina {page_index + 1}, intrebarea {number}: coloanele Corect nu au fost identificate sigur.")
                    options = [t for _, t in answer_cells]
                    boxes = [qbox] + [b for b, _ in answer_cells]
                    visuals = [_is_visual(b, images, drawings, fitz) for b in boxes]
                    if not text and not visuals[0]:
                        raise ImportFailure(f"Pagina {page_index + 1}, intrebarea {number}: enunt gol.")
                    for index, option in enumerate(options):
                        if not option and not visuals[index + 1]:
                            raise ImportFailure(f"Pagina {page_index + 1}, intrebarea {number}: varianta {index + 1} este goala.")
                    flags = [t.strip() for _, t in key_cells]
                    correct = [i for i, value in enumerate(flags) if value == "1"]
                    gradable = all(value in ("0", "1") for value in flags) and bool(correct)
                    if number <= previous_number:
                        section += 1
                        if number != 1:
                            report["warnings"].append(
                                f"P{page_index + 1}: numerotarea reincepe la {number}; verificati originalul.")
                    elif previous_number and number != previous_number + 1:
                        report["warnings"].append(
                            f"P{page_index + 1}: salt de numerotare {previous_number} -> {number}.")
                    previous_number = number
                    if (section, number) in found_indices:
                        raise ImportFailure("Numar duplicat in aceeasi sectiune.")
                    found_indices.add((section, number))
                    ordinal = len(questions)
                    qid = f"{category}-{digest[:10]}-{ordinal + 1:04d}"
                    warning = ""
                    if not gradable:
                        warning = ("Cheie incompleta sau ambigua in PDF. Intrebarea este pastrata "
                                   "pentru studiu, dar NU primeste punctaj. Verifica originalul.")
                        report["warnings"].append(f"P{page_index + 1}, nr. {number}: {warning}")
                    topic = topic_for(text, options)
                    questions.append({
                        "id": qid, "category": category, "ordinal": ordinal,
                        "sourceFile": f"{category}-{digest[:20]}.pdf",
                        "number": number, "section": section, "page": page_index + 1,
                        "text": text, "options": options, "correct": correct,
                        "gradable": gradable, "warning": warning, "topic": topic,
                        "visuals": visuals,
                        "boxes": [[round(float(v), 3) for v in b] for b in boxes],
                    })
                    page_rows.append({"id": qid, "number": number, "section": section,
                                      "flags": flags, "gradable": gradable, "visuals": visuals})
            if not page_rows:
                raise ImportFailure(f"Pagina {page_index + 1}: niciun rand cu intrebari. Import incomplet refuzat.")
            # Cross-check against the independent text layer of the index column.
            # A missed grid row must not silently become a smaller study bank.
            if index_areas:
                left = min(b.x0 for b in index_areas) - 1
                right = max(b.x1 for b in index_areas) + 1
                for word in page.get_text("words"):
                    if not re.fullmatch(r"\d{1,5}", word[4]):
                        continue
                    center = fitz.Point((word[0] + word[2]) / 2, (word[1] + word[3]) / 2)
                    if left < center.x < right and not any(b.contains(center) for b in index_areas):
                        raise ImportFailure(
                            f"Pagina {page_index + 1}: numarul {word[4]} din coloana de index "
                            "nu este inclus intr-un rand extras. Importul partial a fost refuzat.")
            report["pages"].append({"page": page_index + 1, "count": len(page_rows), "rows": page_rows})
    if not questions:
        raise ImportFailure("Nu am gasit intrebari in PDF.")
    report["questionCount"] = len(questions)
    report["gradableCount"] = sum(q["gradable"] for q in questions)
    report["sectionCount"] = section
    return questions, report


def render_cell(pdf: Path, question: dict, index: int) -> bytes:
    if index not in (0, 1, 2, 3):
        raise ValueError("Decupaj invalid.")
    fitz = _fitz()
    with fitz.open(pdf) as doc:
        page = doc[question["page"] - 1]
        b = fitz.Rect(question["boxes"][index])
        # Never include adjacent Corect columns in question/answer previews.
        b = fitz.Rect(b.x0 + .65, b.y0 + .65, b.x1 - .65, b.y1 - .65)
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=b, alpha=False)
        return pix.tobytes("png")
