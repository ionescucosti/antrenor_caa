# Rokid AI RAG Backend

Backend Python/FastAPI care primeste intrebarile extrase de ochelarii Rokid (dupa OCR local) si
raspunde **numai pe baza documentelor si a seturilor de intrebari verificate** incarcate de tine.
Ruleaza pe serverul tau, independent de aplicatia Android Rokid, si este singura sursa de adevar
pentru raspunsurile generate de AI.

Backend-ul si **Antrenor CAA** (aplicatia de studiu) ruleaza in **acelasi proces**: pachetul
`rokid_backend/` include routerul `rokid_backend/caa`, care porteaza handlerul HTTP din `app.py` pe
FastAPI si reutilizeaza neschimbate `Engine`, `importer.py` si interfata din `static/`. Un singur
mediu virtual (`.venv`), un singur `requirements.txt`, un singur port: aplicatia de studiu la `/`,
administrarea backend-ului la `/admin`, endpointul Rokid la `/api/v1/questions/answer`.
`importer.py` este folosit si pentru a transforma PDF-urile ANR in seturi Q&A verificate.

## Ce face

```text
Documente (PDF/DOCX/TXT/MD/CSV/JSON/XLSX) ──> parsare (+OCR) ──> chunking ──> embeddings ──> ChromaDB
Seturi Q&A verificate (JSON/JSONL/CSV/PDF ANR) ──> validare ──> SQLite + embeddings intrebari ──> ChromaDB
                                                                                 │
POST /api/v1/questions/answer ──> validare ──> potrivire exacta Q&A ──> potrivire semantica Q&A ──> RAG + LLM ──> JSON
```

Ordinea de precedenta la raspuns:

1. **Potrivire exacta** cu o intrebare verificata (fara apel LLM; raspunsul corect este mapat pe
   optiunile curente dupa **text**, nu dupa litera, deci optiunile reordonate primesc eticheta corecta).
2. **Potrivire semantica** cu o intrebare verificata (similaritate peste prag, fara conflicte).
3. **RAG** peste fragmentele din documente, cu LLM si iesire structurata validata.

Politica implicita este **fundamentare stricta**: fara dovezi, raspunsul este `insufficient_context`.
Fallback-ul pe cunostinte generale exista, dar este dezactivat (`ROKID_ALLOW_GENERAL_KNOWLEDGE=false`).

## Pornire rapida (local, fara Docker)

```bash
# din radacina proiectului
cp .env.example .env            # editeaza ROKID_ADMIN_PASSWORD, furnizorul LLM, cheia API
python3 porneste.py             # creeaza .venv, instaleaza requirements.txt, porneste pe 8765, deschide browserul
# sau, manual, cu reincarcare la modificari de cod:
.venv/bin/uvicorn rokid_backend.main:app --reload --port 8765
```

Variabile acceptate si fara prefix, pentru compatibilitate cu un `.env` existent: `DATABASE_URL`
(Postgres prin `psycopg`; daca nu raspunde se foloseste SQLite si apare un avertisment in *Stare*),
`OPENAI_API_KEY`, `OPENAI_MODEL`, `ANTHROPIC_API_KEY`. Antrenor CAA isi pastreaza propriul SQLite in
`date/` (`ROKID_CAA_DATA_DIR`); `ROKID_CAA_ENABLED=false` opreste servirea lui, iar
`ROKID_CAA_AUTO_DOWNLOAD=true` descarca PDF-urile ANR lipsa la pornire (porneste.py o face implicit).

Datele backend-ului (SQLite, ChromaDB, fisiere incarcate) ajung in `rokid_data/` (configurabil cu
`ROKID_DATA_DIR`); folderul `date/` ramane al aplicatiei Antrenor CAA.

Deschide `http://127.0.0.1:8765/admin`, autentifica-te cu `ROKID_ADMIN_USERNAME` /
`ROKID_ADMIN_PASSWORD`, apoi:

1. **Stare** -> genereaza un token cu scope `device` pentru ochelari (afisat o singura data).
2. **Documente** -> incarca documentatia. Starea si erorile de parsare apar in lista.
3. **Seturi Q&A** -> incarca `rokid_backend/examples/python_basics_qa.json` sau un PDF ANR (alege categoria C/D).
4. **Test RAG** -> trimite o intrebare cu optiuni si vezi raspunsul, sursele si timpii pe etape.

Pentru dezvoltare fara chei API lasa `ROKID_LLM_PROVIDER=mock` si `ROKID_EMBEDDING_PROVIDER=hash`:
potrivirea exacta/reordonata functioneaza complet; RAG-ul foloseste o euristica lexicala, nu un model.

## Configurare

Toate setarile sunt variabile de mediu cu prefixul `ROKID_` (vezi `.env.example` si
`rokid_backend/core/config.py`). Cele importante:

| Variabila | Rol |
|---|---|
| `ROKID_ADMIN_PASSWORD` | parola interfetei admin (clar sau hash `pbkdf2_sha256$...`) |
| `ROKID_ADMIN_API_TOKEN` | token bearer optional cu scope admin, pentru scripturi |
| `ROKID_DEVICE_TOKENS` | tokenuri initiale pentru ochelari (scope `device`), separate prin virgula |
| `ROKID_EMBEDDING_PROVIDER` | `hash` (offline, dezvoltare), `chroma_default` (MiniLM local), `openai` |
| `ROKID_LLM_PROVIDER` | `mock`, `openai`, `openai_compatible`, `anthropic` |
| `ROKID_LLM_MODEL`, `ROKID_LLM_API_KEY`, `ROKID_LLM_BASE_URL` | modelul si cheia furnizorului |
| `ROKID_ALLOW_GENERAL_KNOWLEDGE` | `true` permite raspuns fara dovezi (marcat explicit); implicit `false` |
| `ROKID_SEMANTIC_QUESTION_THRESHOLD` | similaritatea minima pentru a folosi direct o intrebare verificata (0.90) |
| `ROKID_OPTION_MATCH_THRESHOLD` | similaritate textuala minima la maparea optiunilor (0.86) |
| `ROKID_MAX_UPLOAD_MB`, `ROKID_RATE_LIMIT_PER_MINUTE` | limite |

Furnizori LLM suportati end-to-end: **Anthropic** (Messages API cu tool-use fortat pentru JSON) si
**OpenAI / compatibil OpenAI** (Chat Completions cu `response_format`; prin `openai_compatible` merg
si Gemini (endpointul compatibil OpenAI), Ollama, vLLM, OpenRouter). Adaugarea altui furnizor
inseamna o clasa noua in `rokid_backend/llm/` cu metoda `generate(system, user, schema) -> dict`; contractul API
nu se schimba. Cheile nu se pun niciodata in aplicatia Rokid.

> Un abonament ChatGPT Plus **nu** include acces la API-ul OpenAI; este nevoie de o cheie API separata.

Important: schimbarea furnizorului de embeddings necesita **reindexarea** documentelor si a
seturilor (vectorii nu sunt compatibili intre modele). Fiecare document retine furnizorul cu care a fost
indexat (`embedding_provider`).

## Seturi Q&A verificate

Format JSON (sau JSONL, CSV cu coloanele `question, option_a..option_x, correct_answer, explanation`):

```json
{"question": "Which Python type is immutable?", "question_type": "multiple_choice",
 "options": [{"label": "A", "text": "List"}, {"label": "B", "text": "Dictionary"}, {"label": "C", "text": "Tuple"}],
 "correct_answer": ["C"], "explanation": "Tuples are immutable."}
```

- `correct_answer` accepta una sau mai multe etichete, sau direct textul optiunii.
- Intrebarile deschise: `"question_type": "open", "correct_answer": "3"`.
- PDF-urile ANR (Lista intrebari clasa C/D) se incarca cu `anr_category=C|D`, sau se preiau cu un clic din
  Antrenor CAA (butonul *Preia PDF-urile ANR din Antrenor CAA*, endpoint `POST /qa-datasets/sync-caa`); cheia 0/1 din coloanele
  *Corect* devine raspunsul verificat, iar sursa retine pagina si numarul intrebarii.
- Inregistrarile invalide sunt raportate individual (set cu starea `indexed_with_errors`); duplicatele
  sunt ignorate; textul optiunilor corecte este pastrat, nu doar litera.

## Unde stau datele

| Date | Loc |
|---|---|
| metadate backend (baze de cunostinte, documente, seturi, inregistrari Q&A, joburi, tokenuri) | Postgres (`DATABASE_URL`) sau SQLite `rokid_data/rokid.sqlite3` |
| embeddings (vectorii fragmentelor si ai intrebarilor) | **ChromaDB**, baza separata de vectori, fisiere in `rokid_data/chroma` |
| fisierele incarcate | `rokid_data/uploads/<kb>/` |
| progresul Antrenor CAA | `date/progres.sqlite3` (separat) |

Postgres nu pastreaza vectori; daca vrei totul intr-un singur server de baze de date, varianta ar fi
pgvector, dar pentru un singur utilizator ChromaDB local este mai simplu si suficient.

## Structura

```text
(radacina proiectului)
  app.py, engine.py, importer.py, static/, date/   Antrenor CAA (neschimbat), porneste.py = launcher unic
  rokid_backend/
    caa/           routerul FastAPI care serveste Antrenor CAA (UI la /, API la /api/*)
    api/           rute FastAPI (questions, documents, qa-datasets, knowledge-bases, admin, health)
    core/          config, securitate (tokenuri, sesiuni, rate limit), text, timing, logging
    ingestion/     parsere (PDF+OCR, DOCX, TXT, MD, CSV, JSON, XLSX), chunking, serviciu documente
    qa/            seturi Q&A: parsare/validare, indexare, potrivire exacta/semantica, mapare optiuni
    rag/           embeddings, ChromaDB, prompturi, pipeline-ul de raspuns
    llm/           abstractie LLM, mock, OpenAI(-compatibil), Anthropic
    models/        SQLAlchemy (SQLite): baze de cunostinte, documente, seturi, inregistrari, joburi, tokenuri
    repositories/  sesiuni DB, serviciu baze de cunostinte
    schemas/       Pydantic (contractul API)
    web/static/    interfata admin (HTML/CSS/JS, fara framework)
    examples/      set Q&A si document de exemplu
  tests/           unittest Antrenor CAA (test_core.py) + tests/rokid/ (pytest, LLM mock, ChromaDB temporar)
  docs/rokid-backend/  aceasta documentatie
  Dockerfile, compose.yaml (cloudflared/caddy), Caddyfile, deploy/systemd/, scripts/tunnel.sh, .env.example, requirements.txt (comune)
```

## Teste

```bash
.venv/bin/python -m pytest -q              # ambele suite (Antrenor CAA + Rokid backend)
.venv/bin/python -m pytest -q tests/rokid  # doar backend-ul
```

Testele sunt deterministe (LLM mock, embeddings hash, ChromaDB intr-un folder temporar). Testul cu un
furnizor real ruleaza doar explicit: `ROKID_SMOKE_LLM=1 ROKID_LLM_PROVIDER=anthropic ROKID_LLM_API_KEY=... pytest tests/rokid/test_smoke_real_llm.py`.

## Securitate (pe scurt)

- Doua domenii de autorizare: `device` (doar `/questions/answer`) si `admin` (tot restul).
- Tokenurile sunt stocate hash-uit (SHA-256), revocabile din interfata; sesiunea admin este un cookie
  HttpOnly semnat HMAC, iar mutatiile prin cookie cer antetul `X-Requested-With: rokid-admin`.
- Limitare de rata per token/IP, limita de marime la upload, verificarea tipului real al fisierului,
  parsare defensiva, loguri JSON fara continutul cererilor sau tokenuri.
- ChromaDB este embedded (nu are port expus); HTTPS public prin Cloudflare Tunnel, fara porturi deschise (vezi `DEPLOYMENT.md`).
- Cererile venite prin tunel sunt recunoscute dupa antetele proxy si NU sunt tratate ca locale: UI-ul de studiu cere sesiune admin de la distanta.
- Dovezile din documente sunt tratate ca date neincredere in prompt, nu ca instructiuni.

Documente conexe: `API_CONTRACT.md`, `DEPLOYMENT.md`, `IMPLEMENTATION_STATUS.md`.
