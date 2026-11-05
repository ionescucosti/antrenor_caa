#!/bin/sh
# Creeaza sau reseteaza rolul "chatbot" (parola "chatbot") si baza de date "chatbot"
# pe serverul PostgreSQL local (instalarea EnterpriseDB din /Library/PostgreSQL/16).
# Necesita parola superuserului "postgres" aleasa la instalare:
#   PGPASSWORD='parola-postgres' sh scripts/fix_postgres_chatbot.sh
set -eu
PSQL=${PSQL:-/Library/PostgreSQL/16/bin/psql}
[ -x "$PSQL" ] || PSQL=psql
HOST=${PGHOST:-127.0.0.1}; PORT=${PGPORT:-5432}
ROLE=${CHATBOT_USER:-chatbot}; PASS=${CHATBOT_PASSWORD:-chatbot}; DB=${CHATBOT_DB:-chatbot}

"$PSQL" -v ON_ERROR_STOP=1 -h "$HOST" -p "$PORT" -U postgres -d postgres -v role="$ROLE" -v pass="$PASS" -v db="$DB" <<'SQL'
SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', :'role', :'pass') WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'role') \gexec
SELECT format('ALTER ROLE %I WITH LOGIN PASSWORD %L', :'role', :'pass') \gexec
SELECT format('CREATE DATABASE %I OWNER %I', :'db', :'role') WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = :'db') \gexec
SELECT format('GRANT ALL PRIVILEGES ON DATABASE %I TO %I', :'db', :'role') \gexec
SQL
"$PSQL" -v ON_ERROR_STOP=1 -h "$HOST" -p "$PORT" -U postgres -d "$DB" -c "GRANT ALL ON SCHEMA public TO \"$ROLE\";"
echo "Rolul si baza exista. Verific autentificarea ca $ROLE:"
PGPASSWORD="$PASS" "$PSQL" -h "$HOST" -p "$PORT" -U "$ROLE" -d "$DB" -c "select current_user, current_database();"
echo "Gata. Reporneste serverul (python3 porneste.py); backend-ul va crea tabelele in Postgres la pornire."
