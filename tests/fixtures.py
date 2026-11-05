from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from importer import extract_bank


def make_pdf(path: Path, *, tiny_column=False, invalid_key=False, category='C',
             trailing_column=False, sliver_bleed=False, key_bleed=False, tip_overlap=False):
    """Synthetic PDF table, resembling ANR column geometry; not ANR content."""
    import pymupdf as fitz
    doc = fitz.open()
    page = doc.new_page(width=842, height=595)
    xs = [12, 34, 264, 401, 423, 544, 565, 694, 716]
    if tiny_column:
        xs.insert(3, 266)
    if sliver_bleed:
        # A 3 pt sliver before the last Corect column, like some ANR rows.
        xs.insert(-1, 697)
    if trailing_column:
        # A stray vertical rule past the last Corect column (empty cell).
        xs.append(756)
    header = ['', 'Intrebare', 'Raspuns 1', 'Corect 1', 'Raspuns 2', 'Corect 2', 'Raspuns 3', 'Corect 3']
    rows = [
        ['1', 'TEST TEHNIC: Ce varianta este marcata cu 1 in aceasta fisa de verificare a aplicatiei?', 'Varianta de test A', '0', 'Varianta de test B', '0', 'Varianta de test C', '1'],
        ['2', 'TEST TEHNIC: Identifica optiunea marcata corect. Figura este pentru verificarea decupajului.', 'Prima optiune', '1', 'A doua optiune', '0', 'A treia optiune', '0'],
        ['1', 'TEST TEHNIC: Selecteaza cele doua optiuni marcate corect.', 'Optiunea unu', '1', 'Optiunea doi', '0', 'Optiunea trei', '1'],
    ]
    if category == 'D':
        rows = rows[:2]
    if invalid_key:
        rows[-1][5] = ''
    if tiny_column:
        for row in [header] + rows:
            row.insert(2, '')
    if sliver_bleed:
        for row in [header] + rows:
            row.insert(-1, '')
    if trailing_column:
        for row in [header] + rows:
            row.append('')
    heights = [40] + [105] * len(rows)
    ys = [35]
    for height in heights:
        ys.append(ys[-1] + height)
    for x in xs:
        page.draw_line((x, ys[0]), (x, ys[-1]), width=.6)
    for y in ys:
        page.draw_line((xs[0], y), (xs[-1], y), width=.6)
    for ri, row in enumerate([header] + rows):
        for ci, text in enumerate(row):
            if not text:
                continue
            box = fitz.Rect(xs[ci] + 2, ys[ri] + 3, xs[ci+1] - 2, ys[ri+1] - 2)
            result = page.insert_textbox(box, text, fontsize=6 if ri == 0 else 8, fontname='helv')
            assert result >= 0, (ri, ci, result)
    # A vector light inside the second question; there are no adjacent key marks.
    page.draw_circle((85, ys[2] + 72), 12, color=(.08,.55,.3), fill=(.08,.55,.3))
    # Glyphs printed across a column rule, as in the ANR originals: the 'Tip'
    # value centred on the question/answer rule, and a hyphen at the end of an
    # answer line that overhangs into the next (sliver or Corect) cell.
    for ri in range(1, len(rows) + 1):
        baseline = ys[ri] + 40
        if tip_overlap:
            width = fitz.get_text_length('11', fontname='helv', fontsize=8)
            page.insert_text((xs[2] - width / 2, baseline), '11', fontsize=8, fontname='helv')
        if key_bleed:
            page.insert_text((xs[5] - 1.9, baseline), '-', fontsize=8, fontname='helv')
        if sliver_bleed:
            page.insert_text((xs[-3] - 0.6, baseline), '-', fontsize=8, fontname='helv')
    doc.save(path)
    doc.close()
    return path


def install_fixture(engine, temp: Path, category='C', **kwargs):
    path = make_pdf(temp / (category + '-TEST.pdf'), category=category, **kwargs)
    questions, report = extract_bank(path, category)
    filename = f"{category}-{report['sha256'][:20]}.pdf"
    (engine.directory / 'sources' / filename).write_bytes(path.read_bytes())
    engine.install_bank(category, questions, report, filename, 'https://example.invalid/TECHNICAL_TEST_ONLY.pdf', 'manual')
    return questions, report
