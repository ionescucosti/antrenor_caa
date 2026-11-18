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
POST /api/v1/questions/answer ──> validare ──> potrivire exacta Q&A ──> potrivire semantica Q&A ──> [documentatie + LLM ‖ cautare web] ──> JSON
```

Ordinea de precedenta la raspuns:

1. **Potrivire exacta** cu o intrebare verificata (fara apel LLM; raspunsul corect este mapat pe
   optiunile curente dupa **text**, nu dupa litera, deci optiunile reordonate primesc eticheta corecta).
   Cand mai multe inregistrari verificate au acelasi enunt (listele ANR refolosesc "Care dintre afirmatiile de
   mai jos este adevarata?" pentru intrebari ale caror optiuni difera printr-un singur cuvant: tacheti /
   turnicheti), ramane doar inregistrarea ale carei optiuni sunt cele de pe ecran; celelalte sunt alte
   intrebari, nu un conflict. Daca inregistrarile verificate chiar dau raspunsuri diferite, se afiseaza toate
   literele (cea mai probabila prima) cu `agreement: conflict` si o nota „verifica: ...”; daca raspunsul verificat nu
   se regaseste printre optiunile citite, se afiseaza textul lui, fara litera, cu o nota.
2. **Potrivire semantica** cu o intrebare verificata (similaritate peste prag, fara conflicte).
3. **Intrebare necunoscuta**: in paralel, raspunsul din **documentatie** (fragmente din documente + LLM, iesire
   structurata validata) si raspunsul prin **cautare web** (OpenAI `web_search`). Daca sunt de acord, raspunsul
   este afisat ca atare (`agreement: consensus`); daca nu, se afiseaza cel de pe internet marcat „Verifica”
   (`disagreement`); daca una din surse nu raspunde la timp, ramane cealalta (`docs_only` / `web_only`). Orice
   raspuns neverificat este pastrat ca **candidat** si poate fi confirmat din admin in setul verificat.

Politica implicita este **fundamentare deschisa** (`ROKID_ANSWER_GROUNDING=open`): dovezile din documente au
prioritate, dar modelul poate decide din cunostintele proprii si spune asta in explicatie. Cu `strict`, fara dovezi
citate raspunsul este `insufficient_context` (comportamentul vechi; `ROKID_ALLOW_GENERAL_KNOWLEDGE=true` permite
atunci un raspuns fara dovezi, marcat explicit). Cifrele masurate sunt in sectiunea „Intrebari necunoscute”.

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
| `ROKID_VISION_MODEL` | modelul care citeste pozele (implicit = modelul LLM); recomandat `gpt-6-luna`, cu `gpt-6.1-terra` pentru poze grele |
| `ROKID_VISION_DETAIL`, `ROKID_VISION_MAX_SIDE` | detaliul imaginii trimis modelului (`high`) si latura maxima dupa redimensionare (1792 px; `0` = fara redimensionare) |
| `ROKID_VISION_REASONING_EFFORT`, `ROKID_LLM_REASONING_EFFORT` | efortul de rationament la GPT-5/GPT-6: `none` la transcriere (cel mai rapid), `low` la RAG; gol = nu se trimite |
| `ROKID_TRANSCRIBE_CONTEXT` | contextul de domeniu din promptul de transcriere (implicit: examenul ANR CAA, 3 optiuni A/B/C); gol = fara |
| `ROKID_VISION_CROP` | `auto` (implicit): ecranul luminos din poza este detectat si trimis decupat la rezolutie completa; `off` = poza intreaga |
| `ROKID_VISION_CROP_MODEL_FALLBACK` | `true` (implicit): cand detectorul nu gaseste ecranul, modelul de vedere este intrebat unde este (un apel mic in plus) |
| `ROKID_VISION_MIN_SHARPNESS` | claritatea minima a decupajului (varianta laplacianului pe o copie de 1200 px; poze clare 120-230, miscate sub 15); implicit 25, 0 = fara filtru |
| `ROKID_LLM_SERVICE_TIER` | optional, nivelul de serviciu OpenAI cu latenta mai mica (`priority`); costa mai mult |
| `ROKID_ANSWER_GROUNDING` | `open` (implicit): documentatia este preferata, dar modelul poate decide din cunostintele proprii; `strict`: doar dovezile incarcate |
| `ROKID_SINGLE_ANSWER` | `true` (implicit): intrebarile au un singur raspuns corect; un raspuns cu mai multe optiuni este tratat ca nesigur |
| `ROKID_ANSWER_CONTEXT` | contextul de domeniu din promptul de raspuns si din cautarea web (implicit: examenul ANR CAA) |
| `ROKID_WEB_SEARCH_ENABLED` | `true` (implicit): cautare web prin unealta `web_search` din OpenAI Responses API, in paralel cu documentatia; doar cu `ROKID_LLM_PROVIDER=openai` |
| `ROKID_WEB_SEARCH_MODEL` | modelul pentru cautarea web (implicit `gpt-6.1-terra`) |
| `ROKID_WEB_SEARCH_TIMEOUT_SECONDS` / `ROKID_DOCS_TIMEOUT_SECONDS` | cat asteapta pipeline-ul raspunsul web (8 s) si pe cel din documentatie (15 s); ramura care intarzie este ignorata |
| `ROKID_WEB_SEARCH_CONTEXT_SIZE` | `low` / `medium` (implicit) / `high`: cat continut citeste modelul din paginile gasite |
| `ROKID_WEB_SEARCH_COUNTRY`, `ROKID_WEB_SEARCH_ALLOWED_DOMAINS`, `ROKID_WEB_SEARCH_BLOCKED_DOMAINS` | localizarea cautarii (implicit `RO`) si liste de domenii separate prin virgula (max 100) |
| `ROKID_WEB_SEARCH_PREFER` | `true` (implicit): la dezacord intre documentatie si internet se afiseaza raspunsul de pe internet, marcat „Verifica” |
| `ROKID_CANDIDATES_ENABLED` | `true` (implicit): raspunsurile neverificate sunt pastrate pentru confirmare in admin > Candidati |

Furnizori LLM suportati end-to-end: **Anthropic** (Messages API cu tool-use fortat pentru JSON) si
**OpenAI / compatibil OpenAI** (Chat Completions cu `response_format`; prin `openai_compatible` merg
si Gemini (endpointul compatibil OpenAI), Ollama, vLLM, OpenRouter). Adaugarea altui furnizor
inseamna o clasa noua in `rokid_backend/llm/` cu metoda `generate(system, user, schema) -> dict`; contractul API
nu se schimba. Cheile nu se pun niciodata in aplicatia Rokid.

> Un abonament ChatGPT Plus **nu** include acces la API-ul OpenAI; este nevoie de o cheie API separata.

Important: schimbarea furnizorului de embeddings necesita **reindexarea** documentelor si a
seturilor (vectorii nu sunt compatibili intre modele). Fiecare document retine furnizorul cu care a fost
indexat (`embedding_provider`).

## Intrebari necunoscute: documentatie + internet

Cand intrebarea nu este intr-un set verificat, pipeline-ul porneste in paralel doua ramuri si le combina:

| Ramura | Ce face | Durata tipica |
|---|---|---|
| documentatie | fragmentele cele mai apropiate din documente (plus inregistrari verificate similare) si LLM-ul (`OPENAI_MODEL`), prompt deschis, o singura optiune | 3 s |
| internet | OpenAI Responses API cu unealta `web_search` (`ROKID_WEB_SEARCH_MODEL`), localizare RO; paginile folosite vin ca surse `kind: web` cu `url` | 5,5 s |

Decizia: acord -> `match_type: rag_web`, `agreement: consensus`; dezacord -> raspunsul de pe internet
(`ROKID_WEB_SEARCH_PREFER=true`), cu explicatia prefixata „Verifica: documentatia indica X.” si `note` in raspunsul
compact; o singura ramura -> `docs_only` / `web_only`; niciuna -> `insufficient_context`. Un raspuns cu mai multe
optiuni la o intrebare cu un singur raspuns corect (`ROKID_SINGLE_ANSWER=true`) este tratat ca nesigur.

Masurat pe 98 de intrebari ANR cu cheie cunoscuta (10 pe tema), cu cele 9 manuale incarcate si fara seturile
verificate: fundamentare stricta 17-27% raspunsuri corecte (restul `insufficient_context`); documentatie plus
cunostintele modelului 92% (gpt-6.1-terra), 95% (gpt-6.1-sol), 84% (gpt-6-luna); cautare web 97-100% (gpt-6.1-terra),
93% (gpt-6-luna). Recomandare: `OPENAI_MODEL=gpt-6.1-sol` pentru documentatie si `ROKID_WEB_SEARCH_MODEL=gpt-6.1-terra`.
O intrebare necunoscuta costa circa 3 centi (apelul web 1 cent plus tokenii). Lantul complet, masurat cu
`scripts/eval_cascade.py` pe aceleasi 98 de intrebari: 98/98 corecte cu web (94 prin consens, 4 prin dezacord rezolvat de
internet), p50 5,8 s; 94/98 fara web (ratate: 3 intrebari de legislatie cu sume din alt act decat cel incarcat, plus una cu
cheie discutabila).

**Candidati.** Fiecare raspuns neverificat este salvat (`candidate_id` in raspuns) si apare in admin, pagina
„Candidati”, cu litera propusa, acordul documentatie/internet si sursele. „Confirma” il muta in setul verificat
„Confirmate din admin” (reindexabil, fara fisier sursa): data viitoare aceeasi intrebare primeste raspunsul verificat
din potrivirea exacta, in cateva milisecunde. „Respinge” il arhiveaza. Mai multe cadre ale aceleiasi intrebari
actualizeaza acelasi candidat (`seen_count`).

## Transcrierea pozelor (ochelari)

Pozele de pe ochelari arata toata camera, iar ecranul cu intrebarea este un dreptunghi mic si luminos in ele: trimisa
intreaga si micsorata, poza ajunge cu text de 4-7 px si modelul fie nu vede nimic, fie inventeaza o intrebare
plauzibila. De aceea, inainte de model: (1) ecranul este detectat dupa luminozitate si decupat la rezolutie
completa (circa 0,25 s), cu modelul ca rezerva pentru localizare; (2) decupajul neclar (miscat) este refuzat cu
mesajul „Poza este neclara”; (3) promptul interzice reconstruirea din memorie si cere doar intrebarea cand optiunile
nu se citesc; in acest caz raspunsul verificat vine ca text (`{"text": "...", "note": "..."}`), fara litera.
Conditia practica: textul optiunilor sa aiba cel putin 13-15 px in poza, adica ecranul pe cel putin jumatate din
latimea cadrului (40-50 cm de un laptop de 14") sau zoom 150% in browser, si capul nemiscat la poza.
`gpt-6.1-terra` citeste textul mic mult mai sigur decat `gpt-6-luna` pe decupaje (recomandat la poze).

Cand cererea aduce o poza (sau textul OCR al ochelarilor nu da raspuns), serverul:

1. **reduce poza** la `ROKID_VISION_MAX_SIDE` px pe latura lunga si o recodeaza JPEG (PyMuPDF): un cadru de 12 MP
   ajunge de la 2-4 MB la cateva sute de KB fara sa piarda textul; pozele cu rotatie EXIF sunt trimise neatinse;
2. **trimite poza modelului de vedere** (`ROKID_VISION_MODEL`) cu promptul de transcriere: intrebarea pe o linie,
   apoi `A.`/`B.`/`C.`, copiate exact, fara numarul intrebarii, cronometru, butoane sau raspunsul marcat;
   `reasoning_effort=none` si `detail=high` tin apelul la 1-3 s;
3. **curata raspunsul** (code fences, "Intrebare:", numar de ordine) si il trece prin acelasi lant ca textul:
   potrivire exacta in seturile verificate (hash pe textul normalizat: fara diacritice, punctuatie, majuscule),
   apoi potrivire semantica prin embeddings (prag 0.90), apoi RAG.

Modelul conteaza doar la pasul 2 si la RAG; pentru intrebarile din seturile ANR raspunsul vine din cheia oficiala,
nu de la model. Parametrii pe care un model ii respinge (`temperature`, `max_tokens`, `reasoning_effort: none`)
sunt eliminati sau coborati automat la primul HTTP 400 si retinuti pentru restul procesului.

Recomandare OpenAI (noiembrie 2026, masurat pe o poza sintetica neclara cu o intrebare ANR, 1792x1008 px):
`gpt-6-luna` la vedere (2,0 s, 2.600 tokeni de imagine, 0,10 $/M; transcriere identica cu originalul),
`gpt-6.1-sol` la raspunsul din documentatie (nu accepta `none`; se coboara automat la `low`) si `gpt-6.1-terra`
la cautarea web (vezi sectiunea anterioara pentru cifre). Treci vederea pe `gpt-6.1-terra` (la vedere a dat acelasi
text in 4,2 s) doar daca pozele reale de pe ochelari ies cu greseli pe `gpt-6-luna`. `gpt-4.1-mini` (2,2 s, 3.350
tokeni de imagine) si `gpt-4o-mini` se opresc pe 14 ianuarie 2027, `gpt-5-mini` pe 11 decembrie 2026.
Embeddings: `text-embedding-3-small` (rapid) sau `text-embedding-3-large` (separa mai bine intrebarile inrudite
din ANR); schimbarea furnizorului sau a modelului cere reindexare, iar pagina *Stare* arata ce mai trebuie reindexat.

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

Evaluarea cap-coada a lantului de raspuns pe intrebari ANR cu cheie cunoscuta (documentele indexate, fara seturile
verificate, intr-o aplicatie temporara; costa apeluri OpenAI reale, circa 3 centi pe intrebare):

```bash
.venv/bin/python scripts/eval_cascade.py                      # gpt-6.1-sol + web gpt-6.1-terra, 10 intrebari pe tema
.venv/bin/python scripts/eval_cascade.py --web none           # doar documentatie + cunostintele modelului
.venv/bin/python scripts/eval_cascade.py --grounding strict   # comportamentul vechi, pentru comparatie
```

Scriptul tipareste corecte/gresite/fara raspuns pe acord (`consensus`, `docs_only`, ...), pe `match_type` si pe tema,
latenta p50/p95 si lista intrebarilor ratate; rezultatele pe intrebare ajung in `rokid_data/eval/`.

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
