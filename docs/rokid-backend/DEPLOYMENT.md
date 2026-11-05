# Instalare in productie, cu HTTPS public prin Cloudflare Tunnel

Serverul este acelasi proces ca local: UI-ul Antrenor CAA la `/`, administrarea la `/admin`, endpointul
Rokid la `/api/v1/questions/answer`. Pe internet il expui prin **Cloudflare Tunnel**: `cloudflared` ruleaza
langa aplicatie, deschide o conexiune de iesire catre Cloudflare, iar Cloudflare serveste domeniul tau cu
HTTPS. Nu deschizi niciun port in firewall, nu ai certificate de administrat, IP-ul serverului nu este expus.

Doua moduri de rulare, cu aceleasi rezultate:

| | Docker (`compose.yaml`) | Fara Docker (systemd, ca local) |
|---|---|---|
| aplicatia | containerul `backend`, port intern 8000 | `.venv/bin/python porneste.py --behind-proxy` pe 127.0.0.1:8765 |
| tunelul | containerul `cloudflared` | `cloudflared` instalat ca serviciu |
| datele | volumul `rokid_data` (`/data`, `/data/caa`) | `rokid_data/` si `date/` din folderul proiectului |
| OCR Tesseract | inclus in imagine | `apt install tesseract-ocr tesseract-ocr-ron` |

## 1. Pregatirea tunelului in Cloudflare (o singura data)

1. Domeniul trebuie sa fie pe Cloudflare (DNS gestionat de Cloudflare; planul gratuit ajunge).
2. [Cloudflare Zero Trust](https://one.dash.cloudflare.com) > **Networks > Tunnels > Create a tunnel** >
   *Cloudflared* > nume (de ex. `antrenor-caa`).
3. Pe pagina *Install and run a connector* copiaza **tokenul** (sirul lung de dupa `--token`). El intra
   in `.env` ca `CLOUDFLARE_TUNNEL_TOKEN=...`. Nu il pune in git.
4. **Public Hostname**: subdomeniul dorit (de ex. `caa.exemplu.ro`), tip `HTTP`, URL:
   - Docker: `backend:8000`
   - systemd: `localhost:8765`
5. Salveaza. Inregistrarea DNS (CNAME catre tunel) este creata automat.

Optional, dar recomandat: **Access > Applications** > *Self-hosted* pentru caile `/admin*` si `/`
(UI-ul de studiu), cu politica „Allow” pe adresa ta de email. Astfel pagina de administrare si aplicatia
de studiu cer autentificare Cloudflare inainte sa ajunga la server. **Nu** proteja `/api/v1/*`: ochelarii
se autentifica doar cu tokenul de dispozitiv.

## 2. Pe server: `.env`

```bash
git clone <repo> /opt/antrenor-caa && cd /opt/antrenor-caa
cp .env.example .env
```

Completeaza in `.env`:

- `ROKID_ENVIRONMENT=production`, `ROKID_COOKIE_SECURE=true` (HTTPS vine de la Cloudflare).
- `ROKID_ADMIN_PASSWORD` (sau hash pbkdf2), `ROKID_DEVICE_TOKENS=<token-lung-aleator>` pentru ochelari.
- `OPENAI_API_KEY`, `OPENAI_MODEL=gpt-6.1-sol` (raspuns din documentatie), `ROKID_WEB_SEARCH_MODEL=gpt-6.1-terra`
  (cautare web) si `ROKID_VISION_MODEL=gpt-6-luna` (poze), sau
  `ROKID_LLM_PROVIDER=anthropic` + `ANTHROPIC_API_KEY`. `ROKID_EMBEDDING_PROVIDER=openai` cu
  `ROKID_EMBEDDING_MODEL=text-embedding-3-small` (acelasi furnizor si model ca local, altfel reindexezi).
- `DATABASE_URL=postgresql+psycopg://...` daca folosesti Postgres; altfel lasa gol (SQLite in `rokid_data/`).
- `CLOUDFLARE_TUNNEL_TOKEN=<tokenul de la pasul 1>`.

## 3a. Pornire cu Docker

```bash
docker compose up -d --build
docker compose logs -f cloudflared      # "Registered tunnel connection" = tunelul este sus
curl https://caa.exemplu.ro/api/v1/health
```

Serviciile din `compose.yaml`: `backend` + `cloudflared` pornesc implicit. `--profile caddy` inlocuieste
tunelul cu Caddy + Let's Encrypt (cere porturile 80/443 si `ROKID_DOMAIN`), iar `--profile quick` deschide
un tunel de test cu URL aleator `*.trycloudflare.com`, fara cont Cloudflare (URL-ul apare in
`docker compose logs cloudflared-quick`).

## 3b. Pornire fara Docker (exact ca local)

```bash
sudo useradd --system --create-home rokid
sudo apt install -y python3 python3-venv tesseract-ocr tesseract-ocr-ron
sudo chown -R rokid:rokid /opt/antrenor-caa
sudo -u rokid python3 porneste.py --no-browser --no-download --behind-proxy   # creeaza .venv, instaleaza; Ctrl+C dupa "started"
sudo cp deploy/systemd/antrenor-caa.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now antrenor-caa
curl -s http://127.0.0.1:8765/api/v1/health
```

Apoi `cloudflared` ca serviciu, cu tokenul tunelului:

```bash
# Ubuntu/Debian
sudo mkdir -p --mode=0755 /usr/share/keyrings
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main" | sudo tee /etc/apt/sources.list.d/cloudflared.list
sudo apt update && sudo apt install -y cloudflared
sudo cloudflared service install <CLOUDFLARE_TUNNEL_TOKEN>
curl https://caa.exemplu.ro/api/v1/health
```

`--behind-proxy` este important: cloudflared trimite cererile de pe 127.0.0.1, iar fara el serverul ar
crede ca orice vizitator de pe internet este „local” si ar deschide UI-ul de studiu fara autentificare.
Serverul refuza oricum ca „local” cererile care poarta antetele `CF-Connecting-IP` / `X-Forwarded-For`.

Test rapid de pe orice masina (inclusiv de pe acest Mac, cu `brew install cloudflared`):
`sh scripts/tunnel.sh` deschide un tunel de test catre serverul local pornit cu `--behind-proxy`;
`sh scripts/tunnel.sh <token>` ruleaza tunelul cu nume in prim-plan.

## 4. Conectarea ochelarilor Rokid

Aplicatia Rokid foloseste orice conexiune la internet (hotspot, Wi-Fi) si apeleaza:

- URL: `https://caa.exemplu.ro/api/v1/questions/answer`
- antet `Authorization: Bearer <device_token>` (tokenul cu scope `device`; nu tokenul admin, nu cheia LLM)
- formular cu `text` si/sau `image`, conform `API_CONTRACT.md`

```bash
curl -s https://caa.exemplu.ro/api/v1/questions/answer \
  -H "Authorization: Bearer <device_token>" \
  -F $'text=Which Python type is immutable?\nA. List\nB. Tuple' -F image=@screenshot.png
```

Cloudflare limiteaza corpul cererii la 100 MB pe planul gratuit, suficient pentru pozele de 12 MB.

## 5. Date, backup, operare

| Ce | Docker | systemd |
|---|---|---|
| SQLite/Chroma/upload-uri, secretul de sesiune | volumul `rokid_data` (`/data`) | `rokid_data/` |
| progresul Antrenor CAA si PDF-urile ANR | `/data/caa` | `date/` |

- Backup Docker: `docker compose stop backend && docker run --rm -v antrenor_caa_rokid_data:/data -v $PWD:/backup alpine tar czf /backup/rokid-data.tgz /data && docker compose start backend`
  (numele volumului este `<folder>_rokid_data`; vezi `docker volume ls`). Fara Docker: arhiveaza `rokid_data/` si `date/`.
- Actualizare: `git pull && docker compose up -d --build`, respectiv `git pull && sudo systemctl restart antrenor-caa`
  (`porneste.py` instaleaza singur dependentele noi).
- Loguri: `docker compose logs backend` / `journalctl -u antrenor-caa -f` (JSON structurat, fara intrebari sau tokenuri).
- Tunelul: `docker compose logs cloudflared` / `journalctl -u cloudflared -f`; starea apare si in Zero Trust > Tunnels.
- Schimbarea furnizorului de embeddings cere reindexarea documentelor si seturilor (buton *Reindexeaza* in admin).
- `/api/docs` (Swagger) este dezactivat cand `ROKID_ENVIRONMENT=production`.

## Rulare locala pentru dezvoltare

`python3 porneste.py` (fara tunel, `http://127.0.0.1:8765`) sau `docker compose -f compose.dev.yaml up --build`
(`http://localhost:8000`, cookie fara flag Secure). Nu folosi varianta de dezvoltare pe internet.
