#!/usr/bin/env python
"""Measure the answer cascade on ANR questions with a known key, exactly as the
server would answer them, but in a throw-away app (own data dir), so nothing
touches the production data.

The documents come from the configured database (the ones already indexed,
minus the question lists), the questions from the Antrenor CAA bank
(date/progres.sqlite3). The verified sets are NOT loaded: every question goes
through the "unknown question" path (documentation + web search in parallel).

    .venv/bin/python scripts/eval_cascade.py                       # gpt-6.1-sol + web gpt-6.1-terra, 10 per topic
    .venv/bin/python scripts/eval_cascade.py --llm gpt-6.1-terra --web none --per-topic 5
    .venv/bin/python scripts/eval_cascade.py --grounding strict    # the old behaviour, for comparison

Costs real OpenAI calls: about 3 cents per question with web search."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import shutil
import sqlite3
import sys
import tempfile
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "rokid"))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine, select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from helpers import admin_headers, make_settings, upload_document  # noqa: E402
from rokid_backend.api.routes_questions import _text_request  # noqa: E402
from rokid_backend.core.config import Settings  # noqa: E402
from rokid_backend.main import create_app  # noqa: E402
from rokid_backend.models import Document  # noqa: E402

SKIP = {"D-9ccd6a5a84-0081", "C-95e9ffc1b8-0288", "D-9ccd6a5a84-0143"}   # bank errors: duplicated option, conflicting keys


def load_documents(env: Settings, exclude: str) -> list[tuple[str, bytes]]:
    engine = create_engine(env.effective_database_url)
    with Session(engine) as session:
        docs = session.execute(select(Document).where(Document.status.in_(("indexed", "indexed_with_errors")))).scalars().all()
        out = []
        for d in docs:
            if exclude and exclude.lower() in d.filename.lower():
                continue
            path = Path(d.stored_path)
            if path.exists():
                out.append((d.filename, path.read_bytes()))
    engine.dispose()
    return out


def load_sample(bank: Path, per_topic: int, seed: int) -> list[dict]:
    con = sqlite3.connect(f"file:{bank}?mode=ro", uri=True)
    by_topic, seen = defaultdict(list), set()
    for (payload,) in con.execute("select payload from questions order by category, ordinal"):
        p = json.loads(payload)
        key = p["text"].strip().lower()
        if p["id"] in SKIP or not p.get("gradable") or key in seen or len(p["options"]) != 3 or len(p["correct"]) != 1:
            continue
        seen.add(key)
        by_topic[p["topic"]].append(p)
    random.seed(seed)
    sample = []
    for topic, items in sorted(by_topic.items()):
        random.shuffle(items)
        sample += items[:per_topic]
    return sample


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llm", default="gpt-6.1-sol", help="model for the documentation answer")
    parser.add_argument("--web", default="gpt-6.1-terra", help="model for the web search; 'none' disables it")
    parser.add_argument("--grounding", default="open", choices=("open", "strict"))
    parser.add_argument("--per-topic", type=int, default=10)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--exclude", default="Lista intrebari", help="skip documents whose name contains this (question lists)")
    parser.add_argument("--bank", default="", help="Antrenor CAA database (default: <caa_data_dir>/progres.sqlite3)")
    parser.add_argument("--out", default="", help="JSON with per-question results (default: rokid_data/eval/cascade_<time>.json)")
    args = parser.parse_args()

    env = Settings()
    key = env.llm_key
    if not key:
        print("OPENAI_API_KEY lipseste din .env", file=sys.stderr)
        return 2
    bank = Path(args.bank) if args.bank else env.caa_data_dir / "progres.sqlite3"
    documents = load_documents(env, args.exclude)
    sample = load_sample(bank, args.per_topic, args.seed)
    if not documents or not sample:
        print(f"documente: {len(documents)}, intrebari: {len(sample)} - nimic de evaluat", file=sys.stderr)
        return 2
    tmp = Path(tempfile.mkdtemp(prefix="rokid_eval_"))
    settings = make_settings(tmp, embedding_provider="openai", embedding_model=env.embedding_model, openai_api_key=key,
                             llm_provider="openai", llm_model=args.llm, llm_reasoning_effort=env.llm_reasoning_effort,
                             answer_grounding=args.grounding, web_search_enabled=args.web != "none", web_search_model=args.web,
                             web_search_timeout_seconds=env.web_search_timeout_seconds, answer_context=env.answer_context,
                             rate_limit_per_minute=10**6, rate_limit_burst=10**6, log_level="ERROR", caa_enabled=False, ocr_enabled=False)
    app = create_app(settings)
    print(f"evaluare: llm {args.llm}, web {args.web}, fundamentare {args.grounding}, {len(documents)} documente, {len(sample)} intrebari", flush=True)
    try:
        with TestClient(app) as client:
            ids = [upload_document(client, name, data).json()["document"]["id"] for name, data in documents]
            states = [client.get(f"/api/v1/documents/{i}", headers=admin_headers()).json() for i in ids]
            failed = [s["filename"] for s in states if s["status"] == "failed"]
            if failed:
                print("documente neindexate:", failed, file=sys.stderr)
                print("prima eroare:", next(s["error"] for s in states if s["status"] == "failed")[:300], file=sys.stderr)
                return 1
            print(f"indexate: {sum(s.get('chunk_count') or 0 for s in states)} fragmente", flush=True)
            answers = app.state.answers

            def run_one(p):
                started = time.perf_counter()
                text = "\n".join([p["text"]] + [f"{l}. {t}" for l, t in zip("ABC", p["options"])])
                resp = answers.answer(_text_request(f"e-{uuid.uuid4().hex[:8]}", "default", text))
                labels = ",".join(resp.answer.option_labels) if resp.answer else ""
                expected = "ABC"[p["correct"][0]]
                verdict = "correct" if labels == expected else ("wrong" if resp.status == "answered" and labels else ("no_answer" if resp.status != "error" else "error"))
                return dict(id=p["id"], topic=p["topic"], question=p["text"], expected=expected, labels=labels, status=resp.status,
                            match_type=resp.match_type, agreement=resp.agreement, verdict=verdict, message=(resp.message or "")[:200],
                            ms=round((time.perf_counter() - started) * 1000), timings=resp.timings_ms,
                            sources=[(s.kind, s.document_name[:60], s.url) for s in resp.sources][:8],
                            explanation=(resp.answer.explanation if resp.answer else "")[:200])

            started = time.time()
            with cf.ThreadPoolExecutor(args.workers) as pool:
                results = list(pool.map(run_one, sample))
            stats = Counter(r["verdict"] for r in results)
            n = len(results)
            lat = sorted(r["ms"] for r in results)
            print(f"\n== {args.llm} + web {args.web} ({args.grounding}): corecte {stats['correct']}/{n} ({stats['correct']/n:.0%}), gresite {stats['wrong']}, "
                  f"fara raspuns {stats['no_answer']}, erori {stats['error']} | precizie cand raspunde {stats['correct']/max(1, stats['correct']+stats['wrong']):.0%} | "
                  f"p50 {lat[n//2]} ms, p95 {lat[int(n*0.95)]} ms | {time.time()-started:.0f}s")
            for name, field in (("acord", "agreement"), ("match_type", "match_type"), ("tema", "topic")):
                groups = defaultdict(Counter)
                for r in results:
                    groups[r[field]][r["verdict"]] += 1
                print(f"   pe {name}:")
                for k, c in sorted(groups.items()):
                    print(f"      {k:32s} corecte {c['correct']:3d}  gresite {c['wrong']:3d}  fara {c['no_answer']:3d}  erori {c['error']:3d}")
            for r in results:
                if r["verdict"] != "correct":
                    print(f"   {r['verdict'].upper():9s} {r['id']} dat {r['labels'] or '-'} asteptat {r['expected']} [{r['agreement']}/{r['match_type']}] | "
                          f"{r['question'][:60]} | {(r['message'] or r['explanation'])[:90]}")
            pending = client.get("/api/v1/admin/candidates?status=pending&limit=500", headers=admin_headers()).json()
            print(f"   candidati de confirmat (in aplicatia temporara): {len(pending)}")
            out = Path(args.out) if args.out else env.data_dir / "eval" / f"cascade_{int(time.time())}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"llm": args.llm, "web": args.web, "grounding": args.grounding, "results": results}, ensure_ascii=False, indent=1))
            print("rezultate:", out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
