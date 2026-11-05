#!/bin/sh
# Exemple de apeluri directe la endpointurile serverului local (curl + python pentru afisare).
# Utilizare: sh scripts/test_endpoints.sh [http://127.0.0.1:8765]
set -eu
BASE=${1:-http://127.0.0.1:8765}
cd "$(dirname "$0")/.."
DEV=$(grep '^ROKID_DEVICE_TOKENS=' .env | cut -d= -f2 | cut -d, -f1)
ADMIN_USER=$(grep '^ROKID_ADMIN_USERNAME=' .env | cut -d= -f2); ADMIN_PASS=$(grep '^ROKID_ADMIN_PASSWORD=' .env | cut -d= -f2)
CJ=$(mktemp); trap 'rm -f "$CJ"' EXIT
PY=.venv/bin/python
pretty() { $PY -c 'import sys,json; d=json.load(sys.stdin); print(json.dumps(d, ensure_ascii=False, indent=1)[:1500])'; }

echo "== 1. Health (public; merge si cu Bearer, keep-alive la 45 s)"; curl -s "$BASE/api/v1/health" -H "Authorization: Bearer $DEV" | pretty

echo; echo "== 2. Apelul aplicatiei Rokid (JSON, contractul clientului)"
BODY='{"request_id":"demo-7a5c6b9e","knowledge_base_id":"default","source_mode":"rokid_ocr",
 "question":"Tuturor navelor aflate în largul mării ori pe toate apele învecinate accesibile navelor maritime li se aplică prevederile:",
 "question_type":"multiple_choice",
 "options":[{"label":"A","text":"COLREG-1972."},{"label":"B","text":"R.N.D."},{"label":"C","text":"Codului ISPS."}],
 "client_confidence":0.93}'
curl -s "$BASE/api/v1/questions/answer" -H "Authorization: Bearer $DEV" -H "Content-Type: application/json; charset=utf-8" \
  -H "Accept: application/json" -H "X-Request-Id: demo-7a5c6b9e" -d "$BODY" | pretty

echo; echo "== 3. Acelasi request_id repetat (idempotenta: raspuns identic, fara reprocesare)"
curl -s "$BASE/api/v1/questions/answer" -H "Authorization: Bearer $DEV" -H "Content-Type: application/json" -d "$BODY" | pretty

echo; echo "== 4. Token gresit -> 401; apoi forma scurta cu formular (doar litera si textul)"
curl -s -o /dev/null -w "   HTTP %{http_code}\n" "$BASE/api/v1/questions/answer" -H "Authorization: Bearer gresit" -H "Content-Type: application/json" -d "$BODY"
curl -s "$BASE/api/v1/questions/answer" -H "Authorization: Bearer $DEV" \
  --data-urlencode $'text=Parâmele vegetale și sintetice sunt folosite la confecționarea baloanelor și paietelor?\nA. da\nB. uneori\nC. nu'; echo
if [ -f scripts/sample.png ]; then
  echo "   text OCR incomplet + poza de rezerva:"; curl -s "$BASE/api/v1/questions/answer?full=1" -H "Authorization: Bearer $DEV" -F "text=text ocr incomplet" -F "image=@scripts/sample.png" | pretty
fi

echo; echo "== 5. Login admin (cookie) si listarea seturilor Q&A"
curl -s -c "$CJ" "$BASE/api/v1/admin/login" -H "Content-Type: application/json" -d "{\"username\":\"$ADMIN_USER\",\"password\":\"$ADMIN_PASS\"}" | pretty
curl -s -b "$CJ" "$BASE/api/v1/qa-datasets" | $PY -c 'import sys,json; [print(" -", d["name"], "|", d["status"], "|", d["record_count"], "inregistrari") for d in json.load(sys.stdin)]'

echo; echo "== 6. Stare sistem (admin)"; curl -s -b "$CJ" "$BASE/api/v1/admin/status" | pretty

echo; echo "== 7. Documentatie interactiva: $BASE/api/docs   (OpenAPI JSON: $BASE/api/openapi.json)"
