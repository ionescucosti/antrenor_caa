# Antrenor CAA - C si D

Aplicatie independenta pentru studiu, rulata local in browser. Nu este
aplicatia oficiala ANR. Optiunea **Simulare examen** foloseste numarul de
intrebari si baremul din metodologia ANR, dar rezultatul este unul de
antrenament, nu un calificativ oficial.

## Un singur server: Antrenor CAA + Rokid AI RAG Backend

Din 3 noiembrie 2026 proiectul porneste un singur proces (FastAPI/uvicorn) care serveste:

- aplicatia de studiu Antrenor CAA la `http://127.0.0.1:8765/` (interfata si API-ul ei `/api/*`, neschimbate);
- backend-ul Rokid AI RAG: endpointul pentru ochelari `POST /api/v1/questions/answer`, API-ul de
  administrare si interfata de administrare la `/admin`.

Pornire: `python3 porneste.py` (creeaza `.venv`, instaleaza `requirements.txt`, porneste serverul si
deschide browserul). Alternativ: `.venv/bin/uvicorn rokid_backend.main:app --port 8765`.
Configurarea backend-ului sta in `.env` (vezi `.env.example`); datele lui in `rokid_data/`, iar
progresul Antrenor CAA ramane in `date/`. Documentatia backend-ului: `docs/rokid-backend/`.
Teste pentru tot proiectul: `.venv/bin/python -m pytest -q`.

Acces la aplicatia de studiu: local (127.0.0.1) fara autentificare, ca inainte; de la distanta (prin
reverse proxy) doar dupa autentificare la `/admin`, pentru ca progresul tau sa nu fie public.

## Ce face

- Alegerea categoriei C/D si a tuturor intrebarilor, a unei teme sau a unei
  simulari de examen (26 de intrebari, admis cu minimum 22 corecte).
- Intrebari, variante si ilustratii preluate din PDF-urile originale.
- La o intrebare cu o singura varianta corecta, alegerea este corectata imediat.
  La mai multe variante corecte, selectezi variantele si apesi Verifica.
- Confirmarea repeta textul complet al raspunsului corect, inclusiv cand ai gresit.
  Nu trece automat mai departe: ai timp sa retii raspunsul.
- Salvare imediata in SQLite; reluare de la intrebarea si confirmarea ramase.
- Progres separat pentru C, D si fiecare tema. Schimbarea categoriei nu sterge nimic.
- Resetarea arhiveaza parcurgerea veche si incepe una noua, fara sa stearga istoricul.
- Statistici pe sesiuni, pe parcurgeri si pe categorii finalizate integral.
- Revizualizarea raspunsurilor, filtrarea greselilor, export CSV/JSON, backup complet.
- Interfata adaptabila la ecrane inguste si comenzi A/B/C sau 1/2/3, Enter pentru continuare.

## Instalare si pornire

Necesita Python 3.10+ si un browser actual, de exemplu Edge, Chrome sau Firefox.
Nu necesita Node, un cont online sau o baza de date instalata separat.

**Windows:** extrage complet ZIP-ul si deschide `Porneste_Windows.bat`.
Daca Python lipseste, instaleaza-l de pe https://www.python.org/downloads/ si
bifeaza optiunea pentru PATH. Nu rula aplicatia direct din arhiva.

**Mac/Linux:** in folderul extras, ruleaza `python3 porneste.py` sau
`sh Porneste_Mac_Linux.sh`. Pe unele distributii Linux este necesar si pachetul
sistem pentru modulul Python `venv`.

La prima pornire, `porneste.py` creeaza `.venv` in folderul aplicatiei si
instaleaza PyMuPDF 1.26.7 din PyPI. Nu modifica bibliotecile globale Python.
Internetul este necesar pentru instalarea initiala si descarcarea surselor.
Dupa instalare si importul surselor, studiul functioneaza offline.

Browserul se deschide automat pe `http://127.0.0.1:8765` (implicit).
Serverul asculta numai pe interfata locala a calculatorului, nu in reteaua casei.
Pastreaza terminalul deschis cat studiezi. Pentru oprire, inchide terminalul
sau foloseste Ctrl+C. Inchiderea numai a browserului lasa serverul pornit;
urmatoarea pornire poate redeschide instanta existenta.

Porneste din nou acelasi fisier pentru a reveni. Nu este un fisier EXE
independent: Python este necesar. Codul sursa este inclus si poate fi modificat.

## Importarea bancilor oficiale

Arhiva NU contine PDF-urile oficiale sau o baza de intrebari deja importata.
La prima pornire, aplicatia incearca sa descarce cele doua fisiere de mai jos.
O descarcare este incercata o singura data; eroarea 429 nu produce cereri repetate.

- C: https://portal.rna.ro/wp-content/uploads/2026/07/Lista-intrebari-clasa-C-CAA.pdf
- D: https://portal.rna.ro/wp-content/uploads/2026/07/Lista-intrebari-clasa-D-CAA.pdf
- Pagina ANR: https://portal.rna.ro/personal-navigant/conducator-ambarcatiune-de-agrement/examinare-caa/
  (adresele vechi de tip SiteAssets/Pagini/... raspund 404 din 2026)

Daca ANR nu raspunde, foloseste **Surse & backup > Incarca PDF** pentru
categoria corespunzatoare. Selecteaza originalul cu intrebari, nu manualul
explicativ, o scanare, un export de pe alt site sau o pagina HTML de eroare.

Cheia este extrasa din cele trei coloane `Corect` ale PDF-ului: marcajele
explicite 0/1, nu raspunsuri produse de un model AI. Sunt tratate si intrebarile
cu mai multe variante corecte. O cheie incompleta/ambigua este marcata
**fara punctaj**; nu este transformata intr-un raspuns ghicit sau intr-o greseala.

Fiecare intrebare are trimitere la pagina, blocul de numerotare si numarul ei
in PDF. Numerele se pot relua in document; aplicatia foloseste identificatori
interni distincti. Desenele sunt decupaje din celulele originale, fara coloanele
cu cheia. Poti deschide si decupajele textului original pentru verificare.

Importul verifica tabelele si numerotarea si se opreste la randuri pe care nu
le poate extrage integral. Verifica **Raport import** si sursa originala dupa
prima importare. Aceste verificari automate nu inlocuiesc o confruntare umana
cu documentul si nu certifica actualitatea fiecarei intrebari.

Temele (lumini, semnale sonore, navigatie etc.) sunt etichete pedagogice
atribuite automat din text, nu clasificari oficiale ANR. Pot exista incadrari
imperfecte. Pentru a parcurge toate intrebarile importate foloseste
**Toata categoria**, indiferent de etichetele tematice.

**Simulare examen** (in lista "Ce vrei sa parcurgi?") extrage la intamplare 26
de intrebari cu cheie completa si te declara admis la minimum 22 de raspunsuri
corecte. Cifrele sunt cele pentru clasele C si D din metodologia ANR (Partea a
II-a: 26 de intrebari, 30 de minute, barem 22). Intrebarile vin din toate
temele, proportional cu numarul de intrebari al fiecarei teme din banca
importata si cel putin una din fiecare. Repartizarea urmeaza deci temele
aplicatiei, nu disciplinele oficiale (COLREG/RND 10, marinarie 6, conducere si
manevra 6, prim ajutor 2, legislatie 2), fiindca PDF-urile ANR nu indica
disciplina fiecarei intrebari. O simulare intrerupta se reia cu aceleasi
intrebari; una noua primeste alt set.

Simularea are un cronometru de 30 de minute, afisat deasupra intrebarii (rosu
in ultimele 5 minute). Cand timpul expira, simularea se incheie singura:
raspunsurile trimise dupa aceea nu mai sunt primite, iar intrebarile ramase
fara raspuns se socotesc gresite, ca la examen (poti fi totusi admis daca ai
deja 22 de raspunsuri corecte). Cronometrul este tinut de server si masoara
acelasi timp activ ca statisticile: merge cat timp simularea este deschisa pe
ecran si se opreste la Pauza, la inchiderea paginii sau cat timp fila este
ascunsa; la reluare continui cu timpul ramas.

ANR descrie listele ca orientative, reprezentand doar o parte din intrebarile
de examen. Aplicatia nu poate adauga intrebarile nepublicate. Formularea si
cheia din sursa pot fi vechi, mai ales la legislatie, sanctiuni si primul ajutor.
Studiul acestora nu inlocuieste regulile actuale, pregatirea practica sau
instructiunile serviciilor de urgenta.

## Resetare si reimport de la zero

Pe pagina **Surse & backup**, cardul fiecarei categorii importate are trei
butoane suplimentare, fiecare cu confirmare:

- **Sterge progresul si statisticile**: sterge definitiv parcurgerile,
  sesiunile si raspunsurile categoriei. Urmatoarea pornire incepe de la prima
  intrebare, iar statisticile categoriei pornesc de la zero. Inainte se
  salveaza automat o copie a bazei in `date/backups`. Intrebarile nu se modifica.
- **Reimporta de la ANR (de la 0)**: descarca din nou PDF-ul oficial si
  reincarca toate intrebarile, chiar daca fisierul este identic cu cel importat.
  Progresul si statisticile categoriei se sterg (cu backup automat).
- **Reimporta din PDF local (de la 0)**: la fel, dar din fisierul ales de tine.

Pe pagina **Statistici**, butonul **Sterge tot istoricul (C + D)** goleste
istoricul ambelor categorii, tot cu backup automat. Butonul *De la inceput* din
timpul antrenamentului doar arhiveaza parcurgerea curenta, fara sa stearga nimic.

Butoanele obisnuite *Descarca de la ANR* si *Incarca PDF* raman neschimbate:
un PDF identic nu modifica nimic, iar un PDF diferit inlocuieste banca.

## Cum functioneaza reluarea si statisticile

**Sesiune:** incepe cand pornesti/reiei studiul unei categorii/selectii. Se
incheie la pauza, schimbarea categoriei, inchiderea/reincarcarea paginii sau
finalizarea parcurgerii. Intrarea in ecranul Statistici nu incepe o sesiune noua.
O inchidere neasteptata este marcata ca intrerupere la urmatoarea pornire.

**Parcurgere:** trecerea o data prin intrebarile selectiei. Poate dura mai multe
sesiuni sau zile. Este completa numai dupa parcurgerea tuturor intrebarilor.
Numai o parcurgere completa a selectiei **Toata categoria** se numara la
categoriile parcurse cap-coada; o tema terminata ramane o parcurgere tematica.
O simulare de examen terminata apare in istoric ca Admis sau Respins si este
numarata separat, pe categorii (simulari admise din cele sustinute).

Sunt pastrate inceputul, ultima activitate/finalul, categoria, selectia, numarul
de raspunsuri, corecte, gresite, fara punctaj, procentul si timpul activ estimat.
Procentul este `corecte / (corecte + gresite)`; intrebarile fara cheie completa
nu intra in acest calcul. Timpul activ este o estimare bazata pe semnale periodice
de activitate, nu un cronometru de examen.

Prima alegere conteaza. Un dublu clic sau reincarcarea unei confirmari nu dubleaza
raspunsurile. La revenire, confirmarea ramane vizibila pana apesi Urmatoarea intrebare.
Dupa Pauza, apasa Continua pentru aceeasi selectie. Inchiderea in timpul studiului
permite reluarea automata. Foloseste o singura fila: pornirea unei sesiuni in alta
fila o inchide pe cea veche, pentru a evita doua cursoare concurente.

**De la inceput** cere confirmare si porneste alta parcurgere a selectiei curente.
Istoricul vechi si progresul altor categorii/teme raman salvate. Reimportul exact
al aceluiasi PDF nu reseteaza progresul. Un PDF cu continut binar diferit arhiveaza
parcurgerile active ale categoriei inlocuite; raspunsurile anterioare raman in istoric.

## Unde se salveaza datele

Totul se afla in folderul `date` de langa aplicatie:

```
date/
  progres.sqlite3   raspunsuri, sesiuni, parcurgeri si intrebari importate
  sources/          PDF-uri originale, pastrate si pentru vechile raspunsuri
  reports/          rapoarte de import
  cache/            decupaje ale celulelor PDF
  backups/          ultimele 10 copii automate ale bazei SQLite
```

Progresul nu depinde de cookie-uri sau de browser. Poti folosi alt browser pe
acelasi calculator, cu aceeasi instanta/folder de date. Nu exista sincronizare
cloud, acces implicit de pe telefon sau transfer automat pe alt calculator.
Nu sterge folderul `date` cand actualizezi aplicatia. Datele locale nu sunt criptate.

La pornire se face o copie a bazei existente daca exista raspunsuri. Pentru o
copie portabila cu tot cu surse foloseste **Surse & backup > Descarca backup complet**.
Exportul CSV/JSON este pentru analiza; nu este un backup restaurabil al aplicatiei.

**Restaurare:** opreste complet aplicatia. Pastreaza separat vechiul folder `date`,
apoi inlocuieste-l cu folderul `date` din ZIP-ul de backup. Porneste din nou.
Nu combina baza restaurata cu fisiere `-wal`/`-shm` ramase din vechea instanta.
La mutarea pe alt calculator, reinstaleaza mediul `.venv` acolo prin `porneste.py`;
nu copia un mediu Python intre sisteme diferite.

## Probleme frecvente

- **Python lipseste / versiune prea veche:** instaleaza Python 3.10+ si reporneste launcherul.
- **Instalarea dependentei esueaza:** verifica accesul la PyPI si permisiunile folderului.
  O conexiune institutionala poate restrictiona descarcarile.
- **429 / server ANR indisponibil:** asteapta si descarca manual cele doua originale;
  nu repeta foarte des cererile. Importa-le apoi din interfata.
- **Import refuzat:** nu a fost activata o banca partiala. Consulta eroarea si raportul;
  un format PDF diferit poate necesita adaptarea `importer.py`. Nu forta o cheie ghicita.
- **Pagina nu raspunde:** verifica terminalul. Dupa repornirea serverului, reincarca pagina.
- **Port ocupat de alt program:** ruleaza `python porneste.py --port 8766`.
- **Sesiune reluata in alta fila:** inchide fila veche si reincarca fila pe care o folosesti.
- **Vreau un folder de date separat:** ruleaza `python porneste.py --data-dir "calea/dorita"`.

## Verificari si limite ale acestei livrari

Au trecut 26 de teste automate de logica/import si 12 scenarii de interfata
cu PDF-uri tehnice sintetice. Au fost verificate si imaginile interfetei desktop
si inguste. Vezi `VERIFICARI.txt` si `verificari/browser_report.json`.

**PDF-urile ANR originale nu au fost disponibile ca fisiere in mediul de dezvoltare.**
Au putut fi citite online, dar descarcarea lor binara a esuat. Prin urmare,
importul integral al celor doua originale si descarcarea lor din mediul tau
NU sunt certificate de aceste teste. Verifica prima importare si raportul ei.
Testele nu au fost executate pe un sistem Windows sau macOS nativ.

Datele din `tests/fixtures.py` sunt exclusiv teste software generate temporar,
nu o banca de studiu, si nu sunt incarcate de aplicatie la pornire.

## Pentru modificarea codului

- `app.py`: server local, import, export si protectia cererilor.
- `engine.py`: baza SQLite, progres si statistici.
- `importer.py`: extragerea tabelelor si cheilor din PDF-uri.
- `static/`: interfata HTML/CSS/JavaScript, fara CDN sau biblioteci JavaScript externe.
- `porneste.py`: configurarea izolata si pornirea.

Teste de baza, dupa instalarea dependentei:

```
python -m unittest discover -s tests -v
```

Testul optional `tests/browser_e2e.py` necesita separat Playwright si Chromium.
Modul `--bridge` foloseste DOM local si un adaptor HTTP pentru medii in care
browserul nu poate naviga; nu echivaleaza cu un test complet de navigare nativa.
Rapoartele specifica modul utilizat. Nu este o dependenta pentru utilizator.

PyMuPDF este o dependenta terta, instalata separat, cu propriile conditii de
licentiere. Consulta documentatia proiectului inaintea redistribuirii sau a
integrarii intr-un produs comercial: https://pymupdf.readthedocs.io/
