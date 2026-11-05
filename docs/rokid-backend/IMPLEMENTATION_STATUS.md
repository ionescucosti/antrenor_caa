# Stadiul implementarii — Rokid AI RAG Backend

Data: 3 noiembrie 2026. Mediu de verificare: macOS, Python 3.14 (venv-ul comun al proiectului), ChromaDB 1.5.9, FastAPI 0.162.

## Livrabile

| # | Livrabil | Stare | Observatii |
|---|---|---|---|
| 1 | Aplicatie FastAPI functionala | gata | `rokid_backend/main.py`, factory `create_app` |
| 2 | Pipeline ingestie documente | gata | PDF (text + OCR prin Tesseract daca e instalat), DOCX, TXT, MD, CSV, JSON, XLSX; deduplicare pe hash; versiuni; joburi cu timpi |
| 3 | Integrare ChromaDB | gata | client persistent embedded, colectii `chunks__<kb>` si `questions__<kb>`, cosine |
| 4 | Ingestie seturi Q&A | gata | JSON/JSONL/CSV + PDF ANR prin `importer.py` existent; validare per inregistrare; duplicate ignorate |
| 5 | Potrivire exacta si semantica | gata | hash pe intrebarea normalizata; embeddings pe intrebare + optiuni sortate; mapare dupa text cu praguri |
| 6 | Generare LLM fundamentata | gata | prompt cu dovezi neincredere, JSON validat, etichete/surse verificate fata de cerere |
| 7 | REST API compatibil Rokid | gata | `POST /api/v1/questions/answer` conform contractului; campuri suplimentare optionale |
| 8 | Interfata admin | gata | HTML/JS fara framework: documente, seturi, baze, test RAG, stare, tokenuri |
| 9 | Teste automate | gata | 50 teste deterministe (pytest), 1 test smoke optional cu furnizor real |
| 10 | Docker | gata | Dockerfile (Tesseract inclus), compose cu Cloudflare Tunnel (HTTPS; Caddy ca alternativa), volume, healthcheck, restart |
| 11-14 | README, API_CONTRACT, DEPLOYMENT, IMPLEMENTATION_STATUS | gata | in romana |

## Server unic (3 noiembrie 2026)

Antrenor CAA este servita de acelasi proces (`rokid_backend/caa`), cu teste dedicate
(`tests/rokid/test_caa_merged.py`: UI cu token, import PDF ANR prin API-ul CAA, raspuns, decupaje,
export, granita de acces pentru clienti ne-locali, fallback SQLite la DATABASE_URL invalid).
Variabilele `DATABASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `ANTHROPIC_API_KEY` sunt acceptate fara prefix.

## Criterii de acceptanta

| Criteriu | Verificat prin |
|---|---|
| Upload PDF si alte formate prin interfata | `test_upload_and_index_supported_types` + interfata |
| Indexare in ChromaDB | numarul de vectori verificat in teste |
| Upload set de intrebari verificate | `test_exact_match_answers_without_llm`, `test_anr_pdf_dataset_via_existing_importer` |
| Recunoasterea unei intrebari cunoscute | potrivire exacta fara LLM (`timings_ms` fara `llm`) |
| Eticheta corecta la optiuni reordonate | `test_reordered_options_get_current_label` (inclusiv raspunsuri multiple) |
| Dovezi pentru intrebari noi | `test_unseen_question_answered_from_documents`, `test_pdf_sources_carry_page_numbers` |
| Raspuns prin LLM configurat | furnizori Anthropic si OpenAI implementati; **neverificat live** (vezi mai jos) |
| Referinte la surse | surse `document`/`qa_dataset` cu pagina |
| Abtinere fara dovezi | `test_empty_knowledge_base_abstains_without_llm`, validarea iesirii LLM |
| Contract JSON Rokid | `test_contract_shape_and_echo`, `test_invalid_questions` |
| HTTPS | Cloudflare Tunnel (`cloudflared` in `compose.yaml` / serviciu systemd); Caddy ca alternativa (`--profile caddy`) |
| Restart fara pierderea datelor | volum `/data` persistent (SQLite + Chroma + uploads) |

## Ce NU a fost verificat in acest mediu

- **Apeluri LLM reale**: nu exista chei API in mediul de dezvoltare. Furnizorii Anthropic si OpenAI
  sunt implementati conform documentatiei lor, dar testul live (`tests/rokid/test_smoke_real_llm.py`) nu a
  fost rulat. Ruleaza-l tu cu `ROKID_SMOKE_LLM=1` dupa configurarea cheii.
- **Build-ul Docker si Caddy/TLS**: Docker este instalat local, dar imaginea nu a fost construita si
  rulata in acest mediu. Fisierele sunt scrise conform practicilor standard.
- **OCR**: Tesseract nu este instalat pe masina de dezvoltare; testul pentru PDF scanat verifica
  raportarea corecta a lipsei OCR. In imaginea Docker Tesseract (eng + ron) este inclus.
- **Embeddings reale**: testele folosesc `hash` (offline). `chroma_default` (MiniLM) descarca modelul
  la prima folosire; nu a fost descarcat aici.

## Limitari si decizii

- Reranking nu este implementat (etapa `reranking` este raportata cu 0 ms); pentru un singur utilizator
  cu top-k mic nu aduce beneficii clare. Se poate adauga in `AnswerService` inainte de prompt.
- Furnizorul `mock` are o euristica lexicala pentru RAG, utila doar la dezvoltare.
- Pragurile de similaritate (`semantic_question_threshold`, `option_match_threshold`) sunt calibrate
  conservator; ajusteaza-le dupa modelul de embeddings ales.
- Limitarea de rata este in memorie (o singura instanta).
- Intrebarile ilustrate din PDF-urile ANR (optiuni-imagine) sunt importate cu textul
  `[ilustratie X]`; nu exista inca potrivire pe imagini.
