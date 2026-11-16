"""Transactional local progress, independent of cookies/browser storage."""
from __future__ import annotations

import io
import json
import random
import sqlite3
import threading
import time
import uuid
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class UserError(ValueError):
    pass


def uid() -> str:
    return uuid.uuid4().hex


def js(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


# Exam simulation. The figures are the ones for classes C and D in the ANR
# methodology (Partea a II-a): one test of 26 questions, 30 minutes, passed
# with at least 22 correct answers.
EXAM_TOPIC = "exam"
EXAM_QUESTIONS = 26
EXAM_PASS = 22
EXAM_MINUTES = 30


def exam_required(total: int) -> int:
    """Correct answers needed to pass; scaled down only for a bank smaller than the exam."""
    return -(-total * EXAM_PASS // EXAM_QUESTIONS)


def exam_allocation(counts: dict[str, int], total: int = EXAM_QUESTIONS) -> dict[str, int]:
    """Split the exam questions between topics in proportion to the size of
    each topic, with at least one question per topic (as long as the total
    allows it) and never more than a topic holds."""
    counts = {t: n for t, n in counts.items() if n > 0}
    available = sum(counts.values())
    total = min(total, available)
    if not total:
        return {}
    quota = {t: total * n / available for t, n in counts.items()}
    plan = {t: min(n, max(1, int(quota[t]))) for t, n in counts.items()}
    while sum(plan.values()) < total:
        # The most under-represented topic that still has questions left.
        topic = max((t for t in plan if plan[t] < counts[t]), key=lambda t: (quota[t] - plan[t], t))
        plan[topic] += 1
    while sum(plan.values()) > total:
        # The guaranteed minimum overshot: take from the most over-represented
        # topic, emptying a topic only when there are more topics than questions.
        spare = [t for t in plan if plan[t] > 1] or [t for t in plan if plan[t]]
        topic = min(spare, key=lambda t: (quota[t] - plan[t], t))
        plan[topic] -= 1
    return {t: n for t, n in plan.items() if n}


class Engine:
    def __init__(self, directory: Path):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        for name in ("sources", "reports", "backups", "cache"):
            (self.directory / name).mkdir(exist_ok=True)
        self.path = self.directory / "progres.sqlite3"
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.path, check_same_thread=False, timeout=15)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS banks(
            category TEXT PRIMARY KEY, digest TEXT NOT NULL, source_file TEXT NOT NULL,
            original_url TEXT NOT NULL, origin TEXT NOT NULL, imported_at REAL NOT NULL,
            report TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS questions(
            id TEXT PRIMARY KEY, category TEXT NOT NULL, ordinal INTEGER NOT NULL,
            topic TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS question_order ON questions(category, ordinal);
        CREATE TABLE IF NOT EXISTS runs(
            id TEXT PRIMARY KEY, category TEXT NOT NULL, topic TEXT NOT NULL,
            digest TEXT NOT NULL, question_ids TEXT NOT NULL, cursor INTEGER NOT NULL DEFAULT 0,
            started_at REAL NOT NULL, ended_at REAL, status TEXT NOT NULL DEFAULT 'active');
        CREATE UNIQUE INDEX IF NOT EXISTS one_active_track
            ON runs(category, topic) WHERE status='active';
        CREATE TABLE IF NOT EXISTS sessions(
            id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
            category TEXT NOT NULL, topic TEXT NOT NULL, started_at REAL NOT NULL,
            last_seen REAL NOT NULL, ended_at REAL, active_seconds REAL NOT NULL DEFAULT 0,
            foreground INTEGER NOT NULL DEFAULT 1, status TEXT NOT NULL DEFAULT 'active');
        CREATE UNIQUE INDEX IF NOT EXISTS one_active_session
            ON sessions((1)) WHERE status='active';
        CREATE TABLE IF NOT EXISTS answers(
            run_id TEXT NOT NULL REFERENCES runs(id), question_id TEXT NOT NULL,
            session_id TEXT NOT NULL REFERENCES sessions(id), selected TEXT NOT NULL,
            correct INTEGER, answered_at REAL NOT NULL, snapshot TEXT NOT NULL,
            PRIMARY KEY(run_id, question_id));
        CREATE INDEX IF NOT EXISTS answers_session ON answers(session_id);
        """)
        self.db.commit()
        if self.db.execute("SELECT COUNT(*) FROM answers").fetchone()[0]:
            self.automatic_backup()
        with self.tx():
            # A previous process may have been killed without a final beacon.
            self.db.execute("UPDATE sessions SET status='interrupted', ended_at=last_seen WHERE status='active'")
            self._meta("schema", 1)

    @contextmanager
    def tx(self):
        with self.lock:
            try:
                self.db.execute("BEGIN IMMEDIATE")
                yield
                self.db.commit()
            except Exception:
                self.db.rollback()
                raise

    def _meta(self, key: str, value: Any) -> None:
        self.db.execute("INSERT INTO meta VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, js(value)))

    def _get_meta(self, key: str, default: Any = None) -> Any:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def _bank(self, category: str):
        row = self.db.execute("SELECT * FROM banks WHERE category=?", (category,)).fetchone()
        if not row:
            raise UserError("Mai intai importa PDF-ul categoriei selectate.")
        return row

    def bank_info(self, category: str) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT * FROM banks WHERE category=?", (category,)).fetchone()
            return dict(row) if row else None

    def install_bank(self, category: str, questions: list[dict], report: dict,
                     source_file: str, url: str, origin: str, force: bool = False) -> dict:
        """Install a bank. An identical PDF is a no-op unless force is set, in
        which case the questions are reloaded and the category's active
        progress is archived, so study restarts from the first question."""
        if category not in ("C", "D") or not questions:
            raise UserError("Banca de intrebari invalida.")
        with self.tx():
            old = self.db.execute("SELECT * FROM banks WHERE category=?", (category,)).fetchone()
            if old and old["digest"] == report["sha256"] and not force:
                return {"unchanged": True, "count": len(questions)}
            now = time.time()
            self.db.execute("UPDATE sessions SET status='bank_changed',ended_at=last_seen WHERE category=? AND status='active'", (category,))
            self.db.execute("UPDATE runs SET status='replaced',ended_at=? WHERE category=? AND status='active'", (now, category))
            self.db.execute("DELETE FROM questions WHERE category=?", (category,))
            self.db.executemany("INSERT INTO questions VALUES(?,?,?,?,?)", [
                (q["id"], category, q["ordinal"], q["topic"], js(q)) for q in questions])
            self.db.execute("""INSERT INTO banks VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(category) DO UPDATE SET digest=excluded.digest,source_file=excluded.source_file,
                original_url=excluded.original_url,origin=excluded.origin,imported_at=excluded.imported_at,report=excluded.report""",
                (category, report["sha256"], source_file, url, origin, now, js(report)))
        return {"unchanged": False, "count": len(questions), "warnings": len(report.get("warnings", []))}

    def _question(self, qid: str) -> dict:
        row = self.db.execute("SELECT payload FROM questions WHERE id=?", (qid,)).fetchone()
        if not row:
            row = self.db.execute("SELECT snapshot FROM answers WHERE question_id=? LIMIT 1", (qid,)).fetchone()
        if not row:
            raise UserError("Intrebarea nu mai exista in banca actuala. Reincepe categoria.")
        return json.loads(row[0])

    def question(self, qid: str) -> dict:
        with self.lock:
            return self._question(qid)

    def _counts(self, where: str, args: tuple) -> dict:
        row = self.db.execute(f"""SELECT COUNT(*) answered,
                COALESCE(SUM(correct=1),0) correct,
                COALESCE(SUM(correct=0),0) wrong,
                COALESCE(SUM(correct IS NULL),0) ungraded
                FROM answers WHERE {where}""", args).fetchone()
        result = dict(row)
        graded = result["correct"] + result["wrong"]
        result["graded"] = graded
        result["percentage"] = round(100 * result["correct"] / graded, 1) if graded else None
        return result

    def _run(self, rid: str):
        row = self.db.execute("SELECT * FROM runs WHERE id=?", (rid,)).fetchone()
        if not row:
            raise UserError("Parcurgerea nu a fost gasita.")
        return row

    def _session(self, sid: str, active: bool = False):
        row = self.db.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
        if not row or (active and row["status"] != "active"):
            raise UserError("Sesiunea nu mai este activa. Apasa Reia; progresul salvat ramane intact. Foloseste o singura fila de studiu.")
        return row

    def _touch(self, sid: str, foreground: bool | None = None) -> None:
        row = self._session(sid)
        if row["status"] != "active":
            return
        now = time.time()
        delta = min(30.0, max(0.0, now - row["last_seen"])) if row["foreground"] else 0.0
        fg = row["foreground"] if foreground is None else int(foreground)
        self.db.execute("UPDATE sessions SET last_seen=?,active_seconds=active_seconds+?,foreground=? WHERE id=?", (now, delta, fg, sid))

    def heartbeat(self, sid: str, foreground: bool) -> dict:
        with self.tx():
            row = self._session(sid)
            if row["status"] != "active":
                return {"active": False}
            self._touch(sid, foreground)
            result: dict = {"active": True}
            run = self._run(row["run_id"])
            if run["topic"] == EXAM_TOPIC:
                result["examRemaining"] = self._exam_remaining(run["id"])
            return result

    def _exam_seconds(self, rid: str) -> float:
        """The exam clock: active time of the sessions of the run, including
        what the open session has not accrued yet. Like the active time in the
        statistics, it stops on pause and while the page is hidden."""
        now, used = time.time(), 0.0
        for row in self.db.execute("SELECT status,foreground,last_seen,active_seconds FROM sessions WHERE run_id=?", (rid,)):
            used += row["active_seconds"]
            if row["status"] == "active" and row["foreground"]:
                used += min(30.0, max(0.0, now - row["last_seen"]))
        return used

    def _exam_remaining(self, rid: str) -> float:
        return round(max(0.0, EXAM_MINUTES * 60 - self._exam_seconds(rid)), 1)

    def _exam_timeout(self, run) -> bool:
        """End an exam whose time is up. It counts as a finished simulation;
        the questions left unanswered are not correct, as at the real exam."""
        if run["topic"] != EXAM_TOPIC or run["status"] != "active" or self._exam_seconds(run["id"]) < EXAM_MINUTES * 60:
            return False
        now = time.time()
        self.db.execute("UPDATE runs SET status='completed',ended_at=? WHERE id=?", (now, run["id"]))
        self.db.execute("UPDATE sessions SET status='completed',ended_at=? WHERE run_id=? AND status='active'", (now, run["id"]))
        self._meta("autoResume", False)
        return True

    def expire(self, sid: str) -> dict:
        """The browser countdown reached zero. The server clock decides: an
        exam that still has time left is returned unchanged."""
        with self.tx():
            session = self._session(sid)
            if session["status"] == "active":
                self._touch(sid)
                self._exam_timeout(self._run(session["run_id"]))
        return self.view(sid)

    def _run_summary(self, row) -> dict:
        result = dict(row)
        ids = json.loads(result.pop("question_ids"))
        result["total"] = len(ids)
        result.update(self._counts("run_id=?", (row["id"],)))
        result["fullCategory"] = row["topic"] == "all"
        result["completeCategory"] = row["topic"] == "all" and row["status"] == "completed"
        result["exam"] = row["topic"] == EXAM_TOPIC
        if result["exam"]:
            result["examRequired"] = exam_required(len(ids))
            result["examMinutes"] = EXAM_MINUTES
            # Decided only at the end; an abandoned simulation is neither passed nor failed.
            result["examPassed"] = result["correct"] >= result["examRequired"] if row["status"] == "completed" else None
            result["examRemaining"] = self._exam_remaining(row["id"]) if row["status"] == "active" else None
            result["examTimedOut"] = row["status"] == "completed" and result["answered"] < len(ids)
        result["active_seconds"] = self.db.execute(
            "SELECT COALESCE(SUM(active_seconds),0) FROM sessions WHERE run_id=?", (row["id"],)).fetchone()[0]
        return result

    def _exam_pool(self, category: str) -> dict[str, list[str]]:
        """Questions an exam may draw, by topic. Only questions with a complete
        key, so that every answer of the simulation can be graded."""
        pool: dict[str, list[str]] = {}
        for qid, topic in self.db.execute(
                "SELECT id,topic FROM questions WHERE category=? AND json_extract(payload,'$.gradable') ORDER BY ordinal", (category,)):
            pool.setdefault(topic, []).append(qid)
        return pool

    def _exam_ids(self, category: str) -> list[str]:
        pool = self._exam_pool(category)
        plan = exam_allocation({topic: len(ids) for topic, ids in pool.items()})
        ids = [qid for topic, count in plan.items() for qid in random.sample(pool[topic], count)]
        random.shuffle(ids)
        return ids

    def _new_session(self, category: str, topic: str) -> str:
        bank = self._bank(category)
        self.db.execute("UPDATE sessions SET status='interrupted',ended_at=last_seen WHERE status='active'")
        run = self.db.execute("SELECT * FROM runs WHERE category=? AND topic=? AND status='active'", (category, topic)).fetchone()
        if not run:
            if topic == EXAM_TOPIC:
                ids = self._exam_ids(category)
            else:
                sql = "SELECT id FROM questions WHERE category=?"
                args: list[str] = [category]
                if topic != "all":
                    sql += " AND topic=?"
                    args.append(topic)
                ids = [r[0] for r in self.db.execute(sql + " ORDER BY ordinal", args)]
            if not ids:
                raise UserError("Tema selectata nu contine intrebari.")
            rid = uid()
            self.db.execute("INSERT INTO runs(id,category,topic,digest,question_ids,started_at) VALUES(?,?,?,?,?,?)", (rid, category, topic, bank["digest"], js(ids), time.time()))
        else:
            rid = run["id"]
        sid, now = uid(), time.time()
        self.db.execute("INSERT INTO sessions(id,run_id,category,topic,started_at,last_seen) VALUES(?,?,?,?,?,?)", (sid, rid, category, topic, now, now))
        self._meta("lastCategory", category)
        self._meta("lastTopic", topic)
        self._meta("lastRun", rid)
        self._meta("autoResume", True)
        return sid

    def start(self, category: str, topic: str = "all") -> dict:
        if category not in ("C", "D") or not isinstance(topic, str) or len(topic) > 150:
            raise UserError("Selectie invalida.")
        with self.tx():
            # Accrue the last seconds only for an actually active foreground session.
            old = self.db.execute("SELECT id FROM sessions WHERE status='active'").fetchone()
            if old:
                self._touch(old[0])
            sid = self._new_session(category, topic)
            # An exam paused in its last seconds may have no time left to resume.
            self._exam_timeout(self._run(self._session(sid)["run_id"]))
        return self.view(sid)

    def view(self, sid: str) -> dict:
        with self.lock:
            session = self._session(sid)
            run = self._run(session["run_id"])
            ids = json.loads(run["question_ids"])
            q = self._question(ids[run["cursor"]])
            answer = self.db.execute("SELECT * FROM answers WHERE run_id=? AND question_id=?", (run["id"], q["id"])).fetchone()
            public = {k: v for k, v in q.items() if k not in ("correct", "boxes")}
            public["multiple"] = len(q["correct"]) > 1
            feedback = None
            if answer:
                feedback = {"selected": json.loads(answer["selected"]), "correct": answer["correct"],
                            "correctIndices": q["correct"], "answerTexts": [q["options"][i] for i in q["correct"]],
                            "answered_at": answer["answered_at"], "gradable": q["gradable"]}
            session_summary = dict(session)
            session_summary.update(self._counts("session_id=?", (sid,)))
            return {"session": session_summary, "run": self._run_summary(run),
                    "question": public, "feedback": feedback,
                    "completed": run["status"] == "completed"}

    def answer(self, sid: str, qid: str, selected: list[int]) -> dict:
        if not isinstance(selected, list) or any(type(i) is not int or i not in (0, 1, 2) for i in selected) or len(selected) != len(set(selected)):
            raise UserError("Raspuns invalid.")
        with self.tx():
            session = self._session(sid)
            run = self._run(session["run_id"])
            if self._exam_timeout(run):
                return self.view(sid)  # too late: the answer is not recorded
            ids = json.loads(run["question_ids"])
            if ids[run["cursor"]] != qid:
                raise UserError("Intrebarea s-a schimbat. Reincarca pagina pentru progresul curent.")
            previous = self.db.execute("SELECT 1 FROM answers WHERE run_id=? AND question_id=?", (run["id"], qid)).fetchone()
            if previous:
                # Idempotent retry after a lost HTTP response; first answer wins.
                pass
            else:
                self._session(sid, active=True)
                if run["status"] != "active":
                    raise UserError("Parcurgerea este incheiata.")
                q = self._question(qid)
                if q["gradable"] and (not selected or (len(q["correct"]) == 1 and len(selected) != 1)):
                    raise UserError("Alege raspunsul inainte de verificare.")
                correct = int(set(selected) == set(q["correct"])) if q["gradable"] else None
                self._touch(sid)
                self.db.execute("INSERT INTO answers VALUES(?,?,?,?,?,?,?)", (
                    run["id"], qid, sid, js(sorted(selected)), correct, time.time(), js(q)))
                if self._counts("run_id=?", (run["id"],))["answered"] == len(ids):
                    now = time.time()
                    self.db.execute("UPDATE runs SET status='completed',ended_at=? WHERE id=?", (now, run["id"]))
                    self.db.execute("UPDATE sessions SET status='completed',ended_at=? WHERE id=?", (now, sid))
                    self._meta("autoResume", False)
        return self.view(sid)

    def next_question(self, sid: str) -> dict:
        with self.tx():
            session = self._session(sid)
            run = self._run(session["run_id"])
            if not self._exam_timeout(run) and run["status"] != "completed":
                self._session(sid, active=True)
                if run["status"] != "active":
                    raise UserError("Parcurgerea nu mai este activa.")
                ids = json.loads(run["question_ids"])
                answered = self.db.execute("SELECT 1 FROM answers WHERE run_id=? AND question_id=?", (run["id"], ids[run["cursor"]])).fetchone()
                if not answered:
                    raise UserError("Raspunde la intrebare inainte de a continua.")
                self._touch(sid)
                self.db.execute("UPDATE runs SET cursor=cursor+1 WHERE id=?", (run["id"],))
        return self.view(sid)

    def end(self, sid: str, reason: str = "paused", resume: bool = False, foreground: bool | None = None) -> dict:
        if reason not in ("paused", "closed", "changed"):
            reason = "paused"
        with self.tx():
            row = self._session(sid)
            if row["status"] == "active":
                self._touch(sid, foreground)
                self.db.execute("UPDATE sessions SET status=?,ended_at=? WHERE id=?", (reason, time.time(), sid))
                self._meta("autoResume", bool(resume))
        return {"ok": True}

    def reset(self, sid: str) -> dict:
        with self.tx():
            session = self._session(sid)
            run = self._run(session["run_id"])
            if session["status"] == "active":
                self._touch(sid)
                self.db.execute("UPDATE sessions SET status='reset',ended_at=? WHERE id=?", (time.time(), sid))
            if run["status"] == "active":
                self.db.execute("UPDATE runs SET status='reset',ended_at=? WHERE id=?", (time.time(), run["id"]))
            new_sid = self._new_session(run["category"], run["topic"])
        return self.view(new_sid)

    def reset_category(self, category: str) -> dict:
        """Archive every active run and session of a category. History and
        statistics are kept; the next start begins at question 1."""
        if category not in ("C", "D"):
            raise UserError("Categoria trebuie sa fie C sau D.")
        with self.tx():
            now = time.time()
            self.db.execute("UPDATE sessions SET status='reset',ended_at=last_seen WHERE category=? AND status='active'", (category,))
            archived = self.db.execute("UPDATE runs SET status='reset',ended_at=? WHERE category=? AND status='active'", (now, category)).rowcount
            if self._get_meta("lastCategory") == category:
                self._meta("autoResume", False)
        return {"ok": True, "archivedRuns": archived}

    def delete_history(self, category: str) -> dict:
        """Delete runs, sessions and answers of a category ("all" = both), so
        the statistics start from zero. A database backup is taken first.
        The imported questions are kept."""
        if category not in ("C", "D", "all"):
            raise UserError("Categoria trebuie sa fie C, D sau all.")
        categories = ("C", "D") if category == "all" else (category,)
        self.automatic_backup()
        with self.tx():
            removed = 0
            for cat in categories:
                self.db.execute("DELETE FROM answers WHERE run_id IN (SELECT id FROM runs WHERE category=?)", (cat,))
                self.db.execute("DELETE FROM sessions WHERE category=?", (cat,))
                removed += self.db.execute("DELETE FROM runs WHERE category=?", (cat,)).rowcount
                if self._get_meta("lastCategory") == cat:
                    self._meta("autoResume", False)
                    self.db.execute("DELETE FROM meta WHERE key='lastRun'")
        return {"ok": True, "deletedRuns": removed}

    def state(self) -> dict:
        with self.lock:
            categories = []
            for cat in ("C", "D"):
                row = self.db.execute("SELECT * FROM banks WHERE category=?", (cat,)).fetchone()
                entry: dict = {"category": cat, "count": 0, "topics": [], "tracks": [], "exam": None}
                if row:
                    report = json.loads(row["report"])
                    entry.update({"count": report["questionCount"], "gradableCount": report["gradableCount"],
                                  "pages": report["pageCount"], "imported_at": row["imported_at"],
                                  "warnings": report.get("warnings", []), "origin": row["origin"],
                                  "digest": row["digest"]})
                    entry["topics"] = [dict(r) for r in self.db.execute("SELECT topic,COUNT(*) count FROM questions WHERE category=? GROUP BY topic ORDER BY topic", (cat,))]
                    entry["tracks"] = [self._run_summary(r) for r in self.db.execute("SELECT * FROM runs WHERE category=? AND status='active'", (cat,))]
                    plan = exam_allocation({topic: len(ids) for topic, ids in self._exam_pool(cat).items()})
                    if plan:
                        total = sum(plan.values())
                        entry["exam"] = {"total": total, "required": exam_required(total), "minutes": EXAM_MINUTES,
                                         "topics": [{"topic": topic, "count": plan[topic]} for topic in sorted(plan)]}
                categories.append(entry)
            last_run_id = self._get_meta("lastRun")
            last_run = self.db.execute("SELECT * FROM runs WHERE id=?", (last_run_id,)).fetchone() if last_run_id else None
            resume = bool(self._get_meta("autoResume", False) and last_run and last_run["status"] == "active")
            return {"categories": categories, "lastCategory": self._get_meta("lastCategory", "C"),
                    "lastTopic": self._get_meta("lastTopic", "all"), "autoResume": resume,
                    "dataLocation": str(self.directory), "version": "1.0.0"}

    def statistics(self) -> dict:
        with self.lock:
            sessions = []
            for row in self.db.execute("SELECT * FROM sessions ORDER BY started_at DESC"):
                item = dict(row)
                item.update(self._counts("session_id=?", (row["id"],)))
                sessions.append(item)
            runs = [self._run_summary(r) for r in self.db.execute("SELECT * FROM runs ORDER BY started_at DESC")]
            categories = []
            for category in ("C", "D"):
                counts = self._counts("run_id IN (SELECT id FROM runs WHERE category=?)", (category,))
                full = [r for r in runs if r["category"] == category and r["completeCategory"]]
                percentages = [r["percentage"] for r in full if r["percentage"] is not None]
                exams = [r for r in runs if r["category"] == category and r["exam"] and r["status"] == "completed"]
                categories.append({"category": category, **counts, "completedRuns": len(full),
                                   "bestCompleted": max(percentages) if percentages else None,
                                   "exams": len(exams), "examsPassed": sum(r["examPassed"] for r in exams)})
            return {"sessions": sessions, "runs": runs, "categories": categories}

    def details(self, kind: str, identity: str) -> list[dict]:
        column = "session_id" if kind == "session" else "run_id"
        with self.lock:
            rows = self.db.execute(f"SELECT * FROM answers WHERE {column}=? ORDER BY answered_at", (identity,))
            return [{"selected": json.loads(r["selected"]), "correct": r["correct"],
                     "answered_at": r["answered_at"], "question": json.loads(r["snapshot"])} for r in rows]

    def automatic_backup(self) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        path = self.directory / "backups" / f"progres-{stamp}.sqlite3"
        with self.lock:
            target = sqlite3.connect(path)
            try:
                self.db.backup(target)
            finally:
                target.close()
        backups = sorted((self.directory / "backups").glob("progres-*.sqlite3"))
        for old in backups[:-10]:
            old.unlink(missing_ok=True)
        return path

    def backup_zip(self) -> bytes:
        snapshot = self.automatic_backup()
        output = io.BytesIO()
        with self.lock, zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(snapshot, "date/progres.sqlite3")
            for folder in ("sources", "reports"):
                for file in sorted((self.directory / folder).glob("*")):
                    if file.is_file():
                        archive.write(file, f"date/{folder}/{file.name}")
            archive.writestr("RESTABILIRE.txt", "Inchide complet aplicatia (inclusiv fereastra terminalului).\nPastreaza o copie a folderului date existent, apoi inlocuieste-l cu folderul date din acest ZIP.\nPorneste din nou aplicatia. Nu combina un fisier SQLite restaurat cu fisiere -wal sau -shm vechi.\n")
            archive.writestr("statistici.json", json.dumps(self.statistics(), ensure_ascii=False, indent=2))
        return output.getvalue()

    def close(self) -> None:
        with self.lock:
            self.db.close()
