#!/usr/bin/env bash
# Sobe a API do Diesel S-10 atras do nginx com TLS, numa tacada.
#
#   sudo ./deploy/instalar_api.sh atlas.creditfy.com.br
#
# Configure .env antes de executar (Neon, Redis, JWT, hosts e origens).
#
# Idempotente: rode de novo para trocar dominio, origem do front ou recarregar
# o servico Python. O server_name e reescrito a partir do valor passado, entao nao ha
# placeholder para esquecer de substituir.

set -euo pipefail

DOMINIO="${1:-}"

if [ -z "$DOMINIO" ]; then
  echo "uso: sudo $0 <dominio> (configure as origens em .env)" >&2
  echo "ex:  sudo $0 atlas.creditfy.com.br" >&2
  exit 1
fi
if [ "$#" -ne 1 ]; then
  echo "As origens CORS agora devem ser configuradas em S10_CORS_ORIGINS no .env." >&2
  exit 1
fi
if [[ ! "$DOMINIO" =~ ^[a-zA-Z0-9][a-zA-Z0-9.-]+$ ]]; then
  echo "dominio invalido" >&2
  exit 1
fi
if [ "$(id -u)" -ne 0 ]; then
  echo "precisa de root (nginx, systemd, certbot). Rode com sudo." >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DONO="${SUDO_USER:-root}"
SITE_DISP="/etc/nginx/sites-available/diesel-api"
SITE_HAB="/etc/nginx/sites-enabled/diesel-api"
UNIT="/etc/systemd/system/diesel-api.service"
BACKUP="/root/backup-nginx-$(date +%Y%m%d-%H%M%S)"

echo "==> repo: $ROOT"
echo "==> dominio: $DOMINIO"
echo "==> usuario do servico: $DONO"

# --------------------------------------------------------------- Python ---
if [ ! -f "$ROOT/.env" ]; then
  echo "Configure $ROOT/.env usando .env.example antes do deploy." >&2
  exit 1
fi
if ! command -v certbot >/dev/null; then
  echo "Instale certbot e python3-certbot-nginx antes de publicar autenticacao." >&2
  exit 1
fi
if [ ! -x "$ROOT/.venv-api/bin/python" ]; then
  sudo -u "$DONO" "${PYTHON_BASE:-python3}" -m venv "$ROOT/.venv-api"
fi
sudo -u "$DONO" "$ROOT/.venv-api/bin/python" -m pip install -r "$ROOT/api/requirements.txt"
cd "$ROOT"
sudo -u "$DONO" env S10_ENVIRONMENT=production "$ROOT/.venv-api/bin/python" -m alembic -c api/alembic.ini upgrade head

# ---------------------------------------------------------------- systemd ---
echo "==> instalando o servico"
sed -e "s#/opt/TCC-modelo1#$ROOT#g" \
    -e "s#^User=.*#User=$DONO#" \
    "$ROOT/deploy/diesel-api.service" > "$UNIT"
systemctl daemon-reload
systemctl enable diesel-api >/dev/null
systemctl restart diesel-api

# ------------------------------------------------------------------ nginx ---
echo "==> backup das configs do nginx em $BACKUP"
mkdir -p "$BACKUP"
cp -a /etc/nginx/sites-available "$BACKUP/" 2>/dev/null || true
cp -a /etc/nginx/sites-enabled "$BACKUP/" 2>/dev/null || true

echo "==> escrevendo $SITE_DISP"
# Reescreve a linha do server_name seja qual for o valor atual: sem isso, um
# placeholder esquecido faz o nginx subir sem erro e nunca atender o dominio.
sed -E "s/^([[:space:]]*server_name[[:space:]]+).*/\1${DOMINIO};/" \
    "$ROOT/deploy/nginx-api.conf" > "$SITE_DISP"
ln -sfn "$SITE_DISP" "$SITE_HAB"

# O certbot escreve no primeiro bloco cujo server_name casa. Se nenhum casava,
# ele sequestra o bloco `default` — que serve /var/www/html e responde 404.
if [ -e /etc/nginx/sites-enabled/default ] &&
   grep -qE "^[[:space:]]*server_name[^;]*${DOMINIO//./\\.}" /etc/nginx/sites-available/default 2>/dev/null; then
  echo "==> o bloco 'default' esta com $DOMINIO no server_name (certbot escreveu nele);"
  echo "    desabilitando para nao conflitar. Reverter: ln -s /etc/nginx/sites-available/default /etc/nginx/sites-enabled/"
  rm -f /etc/nginx/sites-enabled/default
fi

echo "==> validando a config"
if ! nginx -t; then
  echo "" >&2
  echo "Config invalida. NADA foi recarregado: o nginx segue com a config antiga." >&2
  echo "Backup das configs anteriores: $BACKUP" >&2
  exit 1
fi
systemctl reload nginx

# ---------------------------------------------------------------- certbot ---
if command -v certbot >/dev/null; then
  echo "==> TLS"
  if [ -d "/etc/letsencrypt/live/$DOMINIO" ]; then
    echo "    certificado ja existe; religando ao bloco correto"
    certbot --nginx -d "$DOMINIO" --reinstall --redirect --non-interactive
  else
    echo "    emitindo certificado (o certbot vai pedir e-mail e aceite dos termos)"
    certbot --nginx -d "$DOMINIO" --redirect
  fi
else
  echo "==> certbot nao instalado; seguindo sem TLS"
  echo "    sudo apt install -y certbot python3-certbot-nginx && sudo certbot --nginx -d $DOMINIO"
fi

# --------------------------------------------------------------- conferir ---
echo
echo "==> conferindo"
echo -n "    API direto (8080): "
curl -fsS http://127.0.0.1:8080/health/ready || { echo "FALHOU"; journalctl -u diesel-api -n 20 --no-pager; exit 1; }
echo
if [ -d "/etc/letsencrypt/live/$DOMINIO" ]; then
  echo -n "    via nginx (443):   "
  curl -fsS --resolve "$DOMINIO:443:127.0.0.1" "https://$DOMINIO/health/ready" || { echo "FALHOU"; exit 1; }
else
  echo -n "    via nginx (80):    "
  curl -fsS -H "Host: $DOMINIO" http://127.0.0.1/health/ready || { echo "FALHOU"; exit 1; }
fi
echo
echo
echo "Pronto."
echo "  login:     https://$DOMINIO/api/v1/auth/login"
echo "  previsao:  https://$DOMINIO/api/v1/forecast (autenticada)"
echo "  historico: https://$DOMINIO/api/v1/history (paginada)"
echo "  health:    https://$DOMINIO/health/ready"
echo
echo "  logs:      journalctl -u diesel-api -f"
echo "  backup:    $BACKUP"
