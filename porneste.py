"""One-time isolated dependency setup, then start the single local server:
Antrenor CAA (study UI) and the Rokid AI RAG Backend, same process, same port."""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import venv
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def ensure_environment() -> Path:
    environment = ROOT / ".venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        print("Prima pornire: creez un mediu Python izolat in .venv.", flush=True)
        venv.EnvBuilder(with_pip=True).create(environment)
    check = subprocess.run([str(python), "-c", "import pymupdf, fastapi, uvicorn, chromadb, sqlalchemy"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if check.returncode:
        print("Instalez dependentele din requirements.txt (necesita internet; nu modific Python-ul global).", flush=True)
        subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(ROOT / "requirements.txt")], check=True)
    return python


def already_running(url: str) -> bool:
    try:
        with urllib.request.urlopen(url + "/health", timeout=2) as response:
            return json.load(response).get("app") == "CAA-Trainer"
    except Exception:
        return False


def open_when_ready(url: str) -> None:
    for _ in range(120):
        time.sleep(0.5)
        if already_running(url):
            webbrowser.open(url)
            return


def main() -> int:
    if sys.version_info < (3, 10):
        print("Este necesar Python 3.10 sau mai nou.")
        return 1
    parser = argparse.ArgumentParser(description="Antrenor CAA + Rokid AI RAG Backend (server local unic)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("ROKID_PORT", "8765")))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-download", action="store_true", help="nu descarca automat PDF-urile ANR lipsa")
    parser.add_argument("--behind-proxy", action="store_true",
                        help="serverul sta in spatele cloudflared/nginx pe aceeasi masina: ia IP-ul clientului din X-Forwarded-For")
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    print("Antrenor CAA + Rokid backend. Verific instalarea locala...", flush=True)
    try:
        python = ensure_environment()
        probe = socket.socket()
        try:
            probe.bind((args.host, args.port))
        except OSError:
            if already_running(url):
                print("Serverul ruleaza deja:", url)
                if not args.no_browser:
                    webbrowser.open(url)
                return 0
            print(f"Portul {args.port} este ocupat. Inchide cealalta aplicatie sau ruleaza cu --port 8766.")
            return 1
        finally:
            probe.close()
        env = dict(os.environ)
        env.setdefault("ROKID_CAA_AUTO_DOWNLOAD", "false" if args.no_download else "true")
        print("\nANTRENOR CAA + ROKID AI RAG BACKEND")
        print("Aplicatia de studiu:   ", url + "/")
        print("Administrare backend:  ", url + "/admin")
        print("Endpoint Rokid:        ", url + "/api/v1/questions/answer")
        print("Lasa aceasta fereastra deschisa. Ctrl+C opreste serverul.\n", flush=True)
        if not args.no_browser:
            threading.Thread(target=open_when_ready, args=(url,), daemon=True).start()
        command = [str(python), "-m", "uvicorn", "rokid_backend.main:app", "--host", args.host, "--port", str(args.port)]
        if args.behind_proxy:
            command += ["--proxy-headers", "--forwarded-allow-ips", "127.0.0.1,::1"]
        return subprocess.call(command, cwd=ROOT, env=env)
    except KeyboardInterrupt:
        return 0
    except (OSError, subprocess.CalledProcessError) as exc:
        print("\nPornirea nu a reusit:", exc)
        print("Extrage proiectul intr-un folder in care poti scrie. Verifica internetul si instalarea Python.")
        print("Nu sterge folderul date: acolo este progresul tau.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
