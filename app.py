"""Local-only HTTP application. Never bind this development server publicly."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import secrets
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from engine import Engine, UserError
from importer import ImportFailure, extract_bank, render_cell

ROOT = Path(__file__).resolve().parent
# ANR moved the portal to WordPress in 2026; the old SiteAssets links return 404.
SOURCE_BASE = "https://portal.rna.ro/wp-content/uploads/2026/07/Lista-intrebari-clasa-"
SOURCE_URLS = {c: SOURCE_BASE + c + "-CAA.pdf" for c in ("C", "D")}
SOURCE_PAGE = "https://portal.rna.ro/personal-navigant/conducator-ambarcatiune-de-agrement/examinare-caa/"
MAX_PDF = 30 * 1024 * 1024


class Application:
    def __init__(self, directory: Path, port: int):
        self.engine = Engine(directory)
        self.port = port
        self.token = secrets.token_urlsafe(32)
        self.jobs = {c: {"status": "idle", "message": ""} for c in ("C", "D")}
        self.jobs_lock = threading.RLock()
        # PyMuPDF is not used simultaneously by multiple threads.
        self.pdf_lock = threading.RLock()

    def set_job(self, category: str, status: str, message: str, **extra) -> None:
        with self.jobs_lock:
            self.jobs[category] = {"status": status, "message": message, **extra}

    def job_state(self) -> dict:
        with self.jobs_lock:
            return {k: dict(v) for k, v in self.jobs.items()}

    def enqueue(self, category: str, data: bytes | None = None, force: bool = False) -> dict:
        if category not in ("C", "D"):
            raise UserError("Categoria trebuie sa fie C sau D.")
        with self.jobs_lock:
            if self.jobs[category]["status"] in ("queued", "downloading", "parsing"):
                raise UserError("Importul acestei categorii este deja in curs.")
            self.set_job(category, "queued", "In asteptare...")
        threading.Thread(target=self._import, args=(category, data, force), daemon=True).start()
        return {"queued": True}

    def _download(self, category: str) -> bytes:
        request = urllib.request.Request(SOURCE_URLS[category], headers={
            "User-Agent": "CAA-Local-Study/1.0", "Accept": "application/pdf"})
        try:
            with urllib.request.urlopen(request, timeout=35) as response:
                data = response.read(MAX_PDF + 1)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise UserError("ANR a raspuns 429 (prea multe cereri). Nu repet automat cererea. Incarca PDF-ul local sau reincearca mai tarziu.") from exc
            raise UserError(f"Descarcarea ANR a esuat: HTTP {exc.code}. Poti incarca manual PDF-ul "
                            f"(cauta 'Lista intrebari clasa {category}' pe {SOURCE_PAGE}).") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise UserError(f"Nu am putut descarca PDF-ul. Verifica internetul sau incarca-l manual. Detaliu: {exc}") from exc
        if len(data) > MAX_PDF:
            raise UserError("PDF prea mare (maximum 30 MB).")
        if b"%PDF-" not in data[:1024]:
            raise UserError("ANR a returnat o pagina web, nu un PDF. Incarca fisierul PDF original.")
        return data

    def _import(self, category: str, data: bytes | None, force: bool = False):
        temp = self.engine.directory / "sources" / f"incoming-{category}-{secrets.token_hex(6)}.pdf"
        try:
            with self.pdf_lock:
                origin = "manual" if data is not None else "anr_download"
                if data is None:
                    self.set_job(category, "downloading", "Se descarca PDF-ul oficial ANR (o singura incercare)...")
                    data = self._download(category)
                temp.write_bytes(data)
                digest = hashlib.sha256(data).hexdigest()
                existing = self.engine.bank_info(category)
                if existing and existing["digest"] == digest and not force:
                    self.set_job(category, "done", "Acest PDF este deja importat. Progresul a ramas neschimbat.")
                    return
                self.set_job(category, "parsing", "Se citesc intrebarile, variantele si coloanele Corect...")
                questions, report = extract_bank(temp, category)
                filename = f"{category}-{digest[:20]}.pdf"
                destination = self.engine.directory / "sources" / filename
                if self.engine.bank_info(category):
                    self.engine.automatic_backup()
                temp.replace(destination)
                report_file = self.engine.directory / "reports" / f"{category}-{digest[:20]}.json"
                report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                result = self.engine.install_bank(category, questions, report, filename, SOURCE_URLS[category], origin, force)
                if force:
                    self.engine.delete_history(category)
                for stale in (self.engine.directory / "cache").glob(f"{category}-*.png"):
                    stale.unlink(missing_ok=True)
                ungraded = len(questions) - report["gradableCount"]
                message = f"{len(questions)} intrebari importate din {report['pageCount']} pagini."
                if force:
                    message = "Reimportat de la zero: " + message + " Progresul si statisticile categoriei au fost sterse (backup in date/backups)."
                if ungraded:
                    message += f" {ungraded} au cheia incompleta si sunt fara punctaj."
                self.set_job(category, "done", message, **result)
        except Exception as exc:
            self.set_job(category, "error", str(exc) or type(exc).__name__)
        finally:
            temp.unlink(missing_ok=True)

    def crop(self, qid: str, index: int) -> bytes:
        if not re.fullmatch(r"[CD]-[0-9a-f]{10}-\d{4,6}", qid):
            raise UserError("Identificator de intrebare invalid.")
        q = self.engine.question(qid)
        bank = self.engine.bank_info(q["category"])
        if bank is None:
            raise UserError("Sursa PDF lipseste.")
        cache = self.engine.directory / "cache" / f"{qid}-{index}.png"
        with self.pdf_lock:
            if not cache.exists():
                source_name = q.get("sourceFile", bank["source_file"])
                if not re.fullmatch(r"[CD]-[0-9a-f]{20}\.pdf", source_name):
                    raise UserError("Sursa invalida.")
                data = render_cell(self.engine.directory / "sources" / source_name, q, index)
                cache.write_bytes(data)
            return cache.read_bytes()


def csv_bytes(items: list[dict], kind: str) -> bytes:
    output = io.StringIO(newline="")
    columns = ["id", "category", "topic", "started_at", "ended_at", "status",
               "answered", "correct", "wrong", "ungraded", "percentage", "active_seconds"]
    headers = ["ID", "Categorie", "Tema", "Inceput (ora locala)", "Sfarsit (ora locala)", "Stare",
               "Raspunsuri", "Corecte", "Gresite", "Fara punctaj", "Procent corecte", "Timp activ (secunde)"]
    if kind == "runs":
        columns += ["total", "completeCategory"]
        headers += ["Total intrebari in parcurgere", "Categorie completa cap-coada"]
    writer = csv.writer(output)
    writer.writerow(headers)
    for row in items:
        values = []
        for col in columns:
            value = row.get(col)
            if col in ("started_at", "ended_at") and value:
                value = datetime.fromtimestamp(value).astimezone().isoformat(timespec="seconds")
            elif col == "active_seconds" and value is not None:
                value = round(value, 1)
            if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
                value = "'" + value
            values.append(value)
        writer.writerow(values)
    return ("\ufeff" + output.getvalue()).encode("utf-8")


def handler_for(app: Application):
    class Handler(BaseHTTPRequestHandler):
        server_version = "CAA-Local/1.0"

        def log_message(self, fmt, *args):
            # Avoid noisy heartbeats; errors are visible in the application.
            if args and str(args[1] if len(args) > 1 else "").startswith("5"):
                print("Server:", fmt % args, file=sys.stderr)

        def check_origin(self):
            allowed = {f"127.0.0.1:{app.port}", f"localhost:{app.port}"}
            if self.headers.get("Host") not in allowed:
                raise UserError("Gazda nepermisa. Foloseste adresa locala afisata in terminal.")
            origin = self.headers.get("Origin")
            if origin and origin not in {f"http://{host}" for host in allowed}:
                raise UserError("Cerere dintr-o origine externa refuzata.")
            if self.headers.get("Sec-Fetch-Site") == "cross-site":
                raise UserError("Cerere externa refuzata.")

        def send(self, body: bytes, content_type: str = "application/json; charset=utf-8", status: int = 200, filename: str | None = None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def json(self, data, status=200):
            self.send(json.dumps(data, ensure_ascii=False).encode("utf-8"), status=status)

        def do_GET(self):
            try:
                self.check_origin()
                parsed = urllib.parse.urlparse(self.path)
                path = parsed.path
                query = urllib.parse.parse_qs(parsed.query)
                if path == "/health":
                    return self.json({"app": "CAA-Trainer", "version": "1.0.0"})
                if path in ("/", "/index.html"):
                    data = (ROOT / "static" / "index.html").read_text(encoding="utf-8").replace("__TOKEN__", app.token)
                    return self.send(data.encode("utf-8"), "text/html; charset=utf-8")
                if path in ("/app.js", "/style.css"):
                    file = ROOT / "static" / path[1:]
                    mime = "text/javascript; charset=utf-8" if path.endswith(".js") else "text/css; charset=utf-8"
                    return self.send(file.read_bytes(), mime)
                if path == "/favicon.ico":
                    return self.send(b"", "image/x-icon", 204)
                if path == "/api/state":
                    return self.json({**app.engine.state(), "jobs": app.job_state()})
                if path == "/api/stats":
                    return self.json(app.engine.statistics())
                if path == "/api/details":
                    return self.json(app.engine.details(query.get("kind", ["run"])[0], query.get("id", [""])[0]))
                if path == "/api/backup":
                    return self.send(app.engine.backup_zip(), "application/zip", filename="Backup_CAA.zip")
                if path == "/api/export":
                    kind = query.get("kind", ["sessions"])[0]
                    if kind not in ("sessions", "runs"):
                        raise UserError("Export invalid.")
                    data = app.engine.statistics()[kind]
                    return self.send(csv_bytes(data, kind), "text/csv; charset=utf-8", filename=f"CAA_{kind}.csv")
                if re.fullmatch(r"/api/report/[CD]", path):
                    bank = app.engine.bank_info(path[-1])
                    if not bank:
                        raise UserError("Categoria nu este importata.")
                    return self.send(bank["report"].encode("utf-8"), filename=f"Raport_import_{path[-1]}.json")
                if re.fullmatch(r"/source/[CD]", path):
                    bank = app.engine.bank_info(path[-1])
                    if not bank:
                        raise UserError("Categoria nu este importata.")
                    source = app.engine.directory / "sources" / bank["source_file"]
                    return self.send(source.read_bytes(), "application/pdf")
                match = re.fullmatch(r"/crop/([CD]-[0-9a-f]{10}-\d{4,6})/([0-3])\.png", path)
                if match:
                    return self.send(app.crop(match.group(1), int(match.group(2))), "image/png")
                return self.json({"error": "Pagina nu exista."}, 404)
            except UserError as exc:
                return self.json({"error": str(exc)}, 400)
            except Exception as exc:
                print("Eroare GET:", type(exc).__name__, str(exc), file=sys.stderr)
                return self.json({"error": "Operatia a esuat. Datele salvate nu au fost sterse. Verifica terminalul."}, 500)

        def do_POST(self):
            try:
                self.check_origin()
                path = urllib.parse.urlparse(self.path).path
                length = int(self.headers.get("Content-Length", "0"))
                limit = MAX_PDF if path.startswith("/api/import/") else 65536
                if length <= 0 or length > limit:
                    return self.json({"error": "Cerere goala sau prea mare (PDF: maximum 30 MB)."}, 413)
                raw = self.rfile.read(length)
                is_pdf = bool(re.fullmatch(r"/api/import/[CD]", path))
                data = {} if is_pdf else json.loads(raw)
                if not isinstance(data, dict):
                    raise UserError("Cerere JSON invalida.")
                token = self.headers.get("X-CAA-Token") or data.get("token", "")
                if not isinstance(token, str) or not secrets.compare_digest(token, app.token):
                    return self.json({"error": "Pagina este veche sau cererea nu este autorizata. Reincarca pagina."}, 403)
                if is_pdf:
                    query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                    return self.json(app.enqueue(path[-1], raw, query.get("force", ["0"])[0] == "1"), 202)
                if path == "/api/download":
                    return self.json(app.enqueue(data.get("category"), None, bool(data.get("force", False))), 202)
                if path == "/api/reset-category":
                    if data.get("wipe", True):
                        return self.json(app.engine.delete_history(data.get("category")))
                    return self.json(app.engine.reset_category(data.get("category")))
                if path == "/api/start":
                    return self.json(app.engine.start(data.get("category"), data.get("topic", "all")))
                if path == "/api/answer":
                    return self.json(app.engine.answer(data.get("sessionId", ""), data.get("questionId", ""), data.get("selected", [])))
                if path == "/api/next":
                    return self.json(app.engine.next_question(data.get("sessionId", "")))
                if path == "/api/end":
                    return self.json(app.engine.end(data.get("sessionId", ""), data.get("reason", "paused"), bool(data.get("resume", False)), data.get("foreground")))
                if path == "/api/reset":
                    return self.json(app.engine.reset(data.get("sessionId", "")))
                if path == "/api/heartbeat":
                    return self.json(app.engine.heartbeat(data.get("sessionId", ""), bool(data.get("foreground", False))))
                return self.json({"error": "Operatie inexistenta."}, 404)
            except (UserError, ImportFailure, json.JSONDecodeError, ValueError) as exc:
                return self.json({"error": str(exc)}, 400)
            except Exception as exc:
                print("Eroare POST:", type(exc).__name__, str(exc), file=sys.stderr)
                return self.json({"error": "Salvarea nu a fost confirmata. NU continua: reincarca pagina si verifica progresul. Vezi terminalul pentru detalii."}, 500)
    return Handler


def main():
    parser = argparse.ArgumentParser(description="Antrenor CAA local, C/D")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "date")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Portul trebuie sa fie intre 1024 si 65535.")
    url = f"http://127.0.0.1:{args.port}"
    try:
        probe = socket.socket()
        probe.bind(("127.0.0.1", args.port))
        probe.close()
    except OSError:
        try:
            with urllib.request.urlopen(url + "/health", timeout=2) as response:
                if json.load(response).get("app") == "CAA-Trainer":
                    print("Aplicatia ruleaza deja:", url)
                    if not args.no_browser:
                        webbrowser.open(url)
                    return
        except Exception:
            pass
        raise SystemExit(f"Portul {args.port} este ocupat. Inchide cealalta aplicatie sau ruleaza cu --port 8766.")
    app = Application(args.data_dir, args.port)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(app))
    server.daemon_threads = True
    print("\nANTRENOR CAA - C / D")
    print("Deschide:", url)
    print("Date salvate in:", app.engine.directory)
    print("Lasa aceasta fereastra deschisa. Ctrl+C opreste serverul.\n", flush=True)
    if not args.no_download:
        for category in ("C", "D"):
            if not app.engine.bank_info(category):
                app.enqueue(category)
    if not args.no_browser:
        threading.Timer(.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=.3)
    except KeyboardInterrupt:
        print("\nAplicatia s-a oprit. Raspunsurile confirmate sunt salvate.")
    finally:
        server.server_close()
        # daemon import workers may still be winding down, so let process exit
        # close the connection rather than racing them during shutdown.


if __name__ == "__main__":
    main()
