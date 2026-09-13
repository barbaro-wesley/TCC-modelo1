#!/usr/bin/env bash
# Prepara a VPS: venv, dependencias, primeira execucao e agendamento.
#
#   ./deploy/instalar_vps.sh                 # venv + deps + primeira execucao
#   ./deploy/instalar_vps.sh --sem-torch     # pula o torch (~800MB); LSTM fica de fora
#   ./deploy/instalar_vps.sh --cron          # instala tambem as linhas no crontab
#   ./deploy/instalar_vps.sh --sem-primeira-execucao
#   PYTHON_BASE=/usr/bin/python3.13 ./deploy/instalar_vps.sh   # outro interpretador
#
# Idempotente: rodar de novo nao duplica linha de cron nem recria a venv.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SEM_TORCH=0
INSTALAR_CRON=0
PRIMEIRA_EXECUCAO=1
for arg in "$@"; do
  case "$arg" in
    --sem-torch) SEM_TORCH=1 ;;
    --cron) INSTALAR_CRON=1 ;;
    --sem-primeira-execucao) PRIMEIRA_EXECUCAO=0 ;;
    *) echo "argumento desconhecido: $arg" >&2; exit 1 ;;
  esac
done

echo "==> repo: $ROOT"

# PYTHON_BASE permite escolher o interpretador que cria a venv (os pins de
# numpy/pandas pedem Python 3.12+, mais novo que o python3 de algumas VPS).
# Nao confundir com PYTHON_BIN do run_semanal.sh, que e o python de execucao.
BASE_PY="${PYTHON_BASE:-$(command -v python3 || true)}"
if [ -z "$BASE_PY" ] || [ ! -x "$BASE_PY" ]; then
  echo "python3 nao encontrado. No Debian/Ubuntu: apt install python3 python3-venv" >&2
  exit 1
fi
echo "==> interpretador base: $BASE_PY ($("$BASE_PY" --version 2>&1))"

# Nao basta o binario existir: quando falta o python3.X-venv, o `python -m venv`
# monta a arvore de diretorios e so depois quebra no ensurepip, deixando uma
# venv sem pip. Testar o pip e o unico jeito de distinguir as duas situacoes.
if [ -x "$ROOT/.venv/bin/python" ] && "$ROOT/.venv/bin/python" -m pip --version >/dev/null 2>&1; then
  echo "==> .venv ja existe"
else
  if [ -d "$ROOT/.venv" ]; then
    echo "==> .venv existente esta incompleta (sem pip); recriando"
    rm -rf "$ROOT/.venv"
  fi
  echo "==> criando .venv"
  if ! "$BASE_PY" -m venv "$ROOT/.venv"; then
    versao="$("$BASE_PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo 3)"
    echo "" >&2
    echo "Falha ao criar a venv. No Debian/Ubuntu falta o pacote:" >&2
    echo "    sudo apt install python${versao}-venv" >&2
    exit 1
  fi
fi
PY="$ROOT/.venv/bin/python"

echo "==> instalando dependencias"
"$PY" -m pip install --upgrade pip >/dev/null
if [ "$SEM_TORCH" -eq 1 ]; then
  # O LSTM so serve de benchmark e perde por ordens de grandeza; sem torch o
  # job precisa rodar com --pular-lstm.
  grep -v '^torch' requirements.txt > /tmp/req-sem-torch.txt
  "$PY" -m pip install -r /tmp/req-sem-torch.txt
  rm -f /tmp/req-sem-torch.txt
  echo "    torch NAO instalado: use sempre --pular-lstm"
else
  "$PY" -m pip install -r requirements.txt
fi

chmod +x "$ROOT/deploy/run_semanal.sh" "$ROOT/deploy/instalar_vps.sh"
mkdir -p "$ROOT/logs" "$ROOT/results/api" "$ROOT/data/raw" "$ROOT/data/processed"

if [ "$PRIMEIRA_EXECUCAO" -eq 1 ]; then
  echo "==> primeira execucao (baixa tudo e treina; pode levar alguns minutos)"
  extra=""
  [ "$SEM_TORCH" -eq 1 ] && extra="--pular-lstm"
  # shellcheck disable=SC2086
  "$ROOT/deploy/run_semanal.sh" --forcar $extra || {
    echo "primeira execucao falhou; veja logs/semanal-$(date +%Y-%m-%d).log" >&2
    exit 1
  }
  echo "==> ok. Previsao em results/api/previsao.json"
fi

if [ "$INSTALAR_CRON" -eq 1 ]; then
  echo "==> instalando cron"
  extra=""
  [ "$SEM_TORCH" -eq 1 ] && extra=" --pular-lstm"
  atual="$(crontab -l 2>/dev/null || true)"
  if echo "$atual" | grep -qF "$ROOT/deploy/run_semanal.sh"; then
    echo "    ja havia linha para run_semanal.sh; crontab nao foi alterado"
  else
    {
      echo "$atual"
      echo "CRON_TZ=America/Sao_Paulo"
      echo "30 9  * * * $ROOT/deploy/run_semanal.sh$extra"
      echo "30 15 * * * $ROOT/deploy/run_semanal.sh$extra"
    } | crontab -
    echo "    duas execucoes diarias (09:30 e 15:30, America/Sao_Paulo)"
  fi
  crontab -l | grep run_semanal.sh || true
fi

echo
echo "Pronto."
echo "  logs:      $ROOT/logs/"
echo "  API le de: $ROOT/results/api/{previsao,historico,status}.json"
echo "  manual:    $ROOT/deploy/run_semanal.sh --forcar"
