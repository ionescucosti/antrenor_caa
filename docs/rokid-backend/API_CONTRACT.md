# Contract API — Rokid AI RAG Backend

Toate rutele sunt sub `/api/v1`. Corpul cererilor si al raspunsurilor este JSON (UTF-8), cu exceptia
incarcarilor de fisiere (`multipart/form-data`). Autentificare: `Authorization: Bearer <token>`.

| Scope | Cine | Ce poate |
|---|---|---|
| `device` | ochelarii Rokid | doar `POST /questions/answer` |
| `admin` | administratorul (token sau sesiune cookie) | tot |

Coduri de eroare HTTP: `401` token lipsa/invalid, `403` scope insuficient (sau antet CSRF lipsa la
sesiune cookie), `413` fisier prea mare, `422` JSON care nu respecta schema, `429` prea multe cereri,
`5xx` eroare interna (corp `{"detail": "..."}`).

## Endpointul principal (consumat de aplicatia Rokid)

Un singur URL, doua forme de cerere. Aplicatia Rokid foloseste forma JSON de mai jos; forma cu formular
(sectiunea urmatoare) este pentru teste manuale din curl, Swagger si pagina *Test RAG* din admin.

```http
POST /api/v1/questions/answer
Authorization: Bearer <device_token>
Content-Type: application/json; charset=utf-8
Accept: application/json
X-Request-Id: <acelasi uuid ca request_id>
```

```json
{
  "request_id": "7a5c6b9e-1b8e-4d1a-9d2c-4d6f0c9a1e33",
  "knowledge_base_id": "default",
  "source_mode": "rokid_ocr",
  "question": "Which Python type is immutable?",
  "question_type": "multiple_choice",
  "options": [{"label": "A", "text": "List"}, {"label": "B", "text": "Dictionary"},
              {"label": "C", "text": "Tuple"}, {"label": "D", "text": "Set"}],
  "raw_ocr_text": "Which Python type is immutable?\nA. List\nB. Dictionary\nC. Tuple\nD. Set",
  "client_confidence": 0.93,
  "image": {"format": "jpeg", "base64": "/9j/4AAQ...", "width": 960, "height": 540}
}
```

| Camp | Obligatoriu | Descriere |
|---|---|---|
| `request_id` | da (1-100 caractere) | reprodus identic in raspuns; cheia de idempotenta (vezi mai jos) |
| `knowledge_base_id` | nu | implicit `default` |
| `question` | da, daca lipseste `image` | textul intrebarii; poate contine si blocul intreg cu `A.`/`B.`/`C.` cand `options` este gol |
| `question_type` | nu | `multiple_choice` (implicit), `multiple_select`, `single_choice`, `true_false`, `open_ended`/`open` |
| `options` | nu | 0-12 elemente `{label, text}`, etichete unice; `[]` la intrebari deschise |
| `raw_ocr_text` | nu | blocul OCR brut; folosit ca sursa pentru optiuni daca `options` este gol |
| `client_confidence` | nu | 0..1, informativ |
| `image` | nu | `{"format","base64","width","height"}`; `base64` poate fi si `data:image/...;base64,...`; max 12 MB decodat |
| `source_mode`, altele | nu | ignorate |

Ordinea de procesare: textul (`question` + `options`, sau blocul brut) este incercat primul: potrivire exacta
sau semantica in seturile verificate, apoi RAG. `image` este transcrisa cu modelul de vedere **doar daca** textul
lipseste sau nu a dat `answered`. Raspuns `200`:

```json
{
  "request_id": "7a5c6b9e-1b8e-4d1a-9d2c-4d6f0c9a1e33",
  "status": "answered",
  "answer": {"option_labels": ["C"], "text": "Tuple", "explanation": "Tuples are immutable."},
  "sources": [{"document_id": "doc_001", "document_name": "python_guide.pdf", "page": 12, "section": null, "kind": "document", "score": 0.83}],
  "processing_time_ms": 412,
  "message": "",
  "match_type": "exact_qa",
  "timings_ms": {"validation": 0.1, "exact_match": 2.4},
  "transcribed_text": null
}
```

Reguli respectate de backend:

- `request_id` este reprodus identic. **Idempotenta**: acelasi `request_id` (pentru acelasi token) primit din nou
  in `ROKID_IDEMPOTENCY_TTL_SECONDS` (implicit 600 s) primeste raspunsul memorat, fara reprocesare; o reincercare
  sosita cat timp prima cerere inca se proceseaza asteapta rezultatul ei.
- Erorile de business vin cu `200` si `status` = `insufficient_context`, `invalid_question` sau `error`
  (`answer` este `null`, `message` explica). Inclusiv JSON care nu respecta schema, daca are `request_id`.
- `401`/`403` token lipsa/gresit sau scope gresit; `413` imagine peste 12 MB (formular); `422` doar pentru corp
  care nu poate fi citit (JSON invalid, fara `request_id`, formular fara `text` si `image`); `429` prea multe cereri.
- Timp: potrivire exacta/semantica sub 100 ms; RAG cu LLM 2-6 s; transcrierea pozei 2-5 s. Timeout-ul LLM
  (`ROKID_LLM_TIMEOUT_SECONDS`, implicit 45) tine totul sub limita de 60 s a clientului.
- `GET /api/v1/health` este public si accepta si antetul Bearer: `200 {"status": "ok", ...}`, exclus de la rate limit.
- Corpul cererii: pana la `ROKID_MAX_BODY_MB` (implicit 32 MB, altfel `413`); poza decodata pana la 12 MB.
  O poza la rezolutie completa (4032x3024, JPEG 90, 1.5-3 MB) are pana la 4 MB in base64 si trece. Cloudflare accepta pana la 100 MB.

### Forma cu formular (teste manuale, admin)

```http
POST /api/v1/questions/answer
Content-Type: multipart/form-data   (sau application/x-www-form-urlencoded cand trimiti doar text)
```

| Camp | Obligatoriu | Descriere |
|---|---|---|
| `text` | unul din `text`/`image` | blocul brut: intrebarea, apoi `A.` / `B.` / `C.` pe linii separate (sau doar intrebarea). Merg si `A)`, `A `, `1)` si forma pe o singura linie |
| `image` | unul din `text`/`image` | poza cu intrebarea (PNG/JPEG/WEBP/GIF, max 12 MB) |
| `request_id` | nu | generat automat daca lipseste |
| `knowledge_base_id` | nu | implicit `default` |

Raspuns implicit compact, doar ce afiseaza HUD-ul: `{"label": "C", "text": "Tuple"}`; `label` lipseste la
intrebari deschise, `"A,C"` la mai multe optiuni corecte; fara raspuns: `{"error": "motivul"}`. Cand documentatia si
internetul nu sunt de acord apare si `"note": "verifica: documentatia si internetul nu sunt de acord"`. Cu `?full=1`
vine acelasi obiect complet ca la forma JSON.

```bash
curl -s http://127.0.0.1:8765/api/v1/questions/answer -H "Authorization: Bearer <device_token>" \
  --data-urlencode $'text=Which Python type is immutable?\nA. List\nB. Tuple'
curl -s http://127.0.0.1:8765/api/v1/questions/answer -H "Authorization: Bearer <device_token>" \
  -F $'text=Which Python type is immutable?\nA. List\nB. Tuple' -F image=@screenshot.png
```

### Campurile raspunsului complet

Acelasi obiect pentru forma JSON si pentru formular cu `?full=1`:

```json
{
  "request_id": "uuid",
  "status": "answered",
  "answer": {"option_labels": ["C"], "text": "Tuple", "explanation": "Tuples are immutable."},
  "sources": [
    {"document_id": "doc_1", "document_name": "python_guide.pdf", "page": 12, "section": "Data types", "kind": "document", "score": 0.83}
  ],
  "processing_time_ms": 412,
  "message": "",
  "match_type": "exact_qa",
  "agreement": "verified",
  "candidate_id": null,
  "timings_ms": {"validation": 0.1, "exact_match": 2.4, "embedding": 0, "semantic_qa": 0, "vector_retrieval": 0, "llm": 0},
  "transcribed_text": null
}
```

| Camp | Semnificatie |
|---|---|
| `status` | `answered`, `insufficient_context`, `invalid_question`, `error` |
| `answer` | `null` pentru orice status diferit de `answered` |
| `answer.option_labels` | etichete **din blocul trimis** (niciodata inventate); `[]` la intrebari deschise |
| `answer.text` | textul optiunii/optiunilor alese sau raspunsul scurt |
| `answer.explanation` | explicatie scurta, in limba intrebarii |
| `sources` | `kind` = `qa_dataset` (raspuns verificat), `document` (documentatie, cu pagina/sectiunea cand exista) sau `web` (cautare web, cu `url`) |
| `match_type` | `exact_qa`, `semantic_qa`, `rag` (documentatie), `rag_web` (documentatie si web de acord), `web`, `general_knowledge` (cunostintele modelului, fara dovezi citate), `none` |
| `agreement` | `verified` (set verificat), `consensus`, `docs_only`, `web_only`, `disagreement` (explicatia incepe cu „Verifica: ...”), `none` |
| `candidate_id` | id-ul candidatului pastrat pentru confirmare in admin, pentru orice raspuns neverificat; altfel `null` |
| `timings_ms` | timpi pe etape: `transcription` (doar cand s-a folosit poza), `validation`, `exact_match`, `embedding`, `semantic_qa`, `parallel` (ramurile in paralel), `docs_ms`, `web_ms`, `vector_retrieval`, `reranking`, `llm`, `answer_validation`, `serialization` |
| `transcribed_text` | textul extras din poza, cand raspunsul a venit din ea; altfel `null` |

Reguli de decizie:

- Intrebare cunoscuta (potrivire exacta): raspunsul verificat este mapat pe optiunile curente dupa
  text; daca maparea este ambigua sau mai multe inregistrari verificate se contrazic, se returneaza
  `insufficient_context` cu explicatie.
- `invalid_question` cand textul este prea scurt, pare deteriorat de OCR, exista o singura optiune,
  optiuni goale sau duplicate, poza nu poate fi citita, ori baza de cunostinte nu exista.
- Intrebare necunoscuta: ramura „documentatie” (fragmente + LLM) si ramura „internet” (`web_search`) ruleaza in
  paralel. Acord -> `rag_web`/`consensus`; dezacord -> raspunsul de pe internet cu `disagreement` si `note`;
  o singura ramura la timp -> `docs_only`/`web_only`; un raspuns cu mai multe optiuni la o intrebare cu un singur
  raspuns corect este tratat ca nesigur. Raspunsurile neverificate primesc `candidate_id`.
- `insufficient_context` cand nicio ramura nu a dat un raspuns sigur (sau, cu `ROKID_ANSWER_GROUNDING=strict`,
  cand nu exista dovezi ori modelul nu a putut cita surse).
- `error` cand furnizorul LLM a esuat sau a returnat o structura invalida.

## Administrare (scope `admin`)

### Baze de cunostinte

- `GET /knowledge-bases` -> lista cu `document_count`, `qa_record_count`.
- `POST /knowledge-bases` `{"id": "python", "name": "Python", "description": ""}` -> `201`.
- `DELETE /knowledge-bases/{id}` -> `204` (sterge documente, seturi si vectorii; `default` nu se sterge).

### Documente

- `POST /documents` (multipart: `file`, `knowledge_base_id`, optional `replace_document_id`) -> `202`
  `{"document": {...}, "duplicate": false, "job_id": "job_...", "message": "..."}`.
  Continut identic deja indexat in aceeasi baza -> `duplicate: true`, fara reindexare.
- `GET /documents?knowledge_base_id=` -> lista (`status`: `pending|processing|indexed|failed`, `error`, `warnings`, `pages`, `chunk_count`, `version`).
- `GET /documents/{id}`, `DELETE /documents/{id}` -> `204`, `POST /documents/{id}/reindex` -> `202`.

### Seturi Q&A

- `POST /qa-datasets` (multipart: `file` json/jsonl/csv/pdf, `knowledge_base_id`, `name`, `anr_category` = `C`/`D` pentru PDF ANR) -> `202`.
- `POST /qa-datasets/sync-caa` (form: `knowledge_base_id`) -> `202` `{"results": [{category, status: queued|unchanged|missing, dataset_id, job_id, message}]}`:
  preia PDF-urile ANR deja importate in Antrenor CAA ca seturi Q&A verificate (idempotent, pe hash).
- `GET /qa-datasets?knowledge_base_id=`, `GET /qa-datasets/{id}`, `GET /qa-datasets/{id}/records?offset=&limit=&q=`.
- `DELETE /qa-datasets/{id}` -> `204`, `POST /qa-datasets/{id}/reindex` -> `202`.

### Admin

- `POST /admin/login` `{"username","password"}` -> cookie `rokid_admin_session` (HttpOnly). Mutatiile
  facute cu cookie necesita antetul `X-Requested-With: rokid-admin`.
- `POST /admin/logout`, `GET /admin/me`.
- `GET /admin/status`: numar documente/seturi pe stare, erori de ingestie, furnizori configurati, `web_search`,
  `answer_grounding`, `candidates_pending`.
- `GET /admin/candidates?knowledge_base_id=&status=pending|confirmed|rejected|all&limit=`: raspunsurile neverificate
  (intrebare, optiuni, litera propusa, `agreement`, `docs_labels`, `web_labels`, surse, `seen_count`).
- `POST /admin/candidates/{id}/confirm` `{"option_labels": ["B"], "explanation": ""}` (corpul este optional) ->
  `{"ok": true, "record_id": "rec_..."}`: creeaza/actualizeaza inregistrarea verificata in setul „Confirmate din admin”.
- `POST /admin/candidates/{id}/reject` -> `{"ok": true}`.
- `GET /admin/jobs?limit=`, `GET /admin/jobs/{id}`: joburi de ingestie cu timpi pe etape.
- `GET /admin/tokens`, `POST /admin/tokens` `{"name","scope":"device|admin"}` (tokenul apare o singura
  data in raspuns), `DELETE /admin/tokens/{id}` (revocare).

### Public

- `GET /health` -> `{"status": "ok|degraded", "version", "components": {...}}` (fara secrete).
- `GET /admin` -> interfata web de administrare. `GET /api/docs` -> OpenAPI (dezactivat in productie).
