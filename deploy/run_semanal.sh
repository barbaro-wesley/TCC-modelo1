#!/usr/bin/env bash
# Wrapper chamado pelo cron: acha a venv, fixa o ambiente, loga em arquivo e
# propaga o codigo de saida. Sem isso o cron roda com PATH minimo e sem
# PYTHONPATH, e o job quebra por motivos que nada tem a ver com o modelo.
#
#   deploy/run_semanal.sh                 # so retreina se houver semana nova
#   deploy/run_semanal.sh --forcar        # retreina de qualquer jeito
#   deploy/run_semanal.sh --pular-lstm    # dispensa o torch
#
# Variaveis opcionais:
#   PYTHON_BIN       caminho do python (default: .venv/bin/python do repo)
#   TZ               fuso do log (default: America/Sao_Paulo)
#   OMP_NUM_THREADS  threads do LightGBM/XGBoost (default: 2, VPS pequena)
#   ALERTA_WEBHOOK   URL que recebe um POST com o motivo quando o job falha

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs
LOG="$ROOT/logs/semanal-$(date +%Y-%m-%d).log"

PY="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
if [ ! -x "$PY" ]; then
  PY="$(command -v python3 || true)"
fi
if [ -z "$PY" ]; then
  echo "python nao encontrado (defina PYTHON_BIN)" | tee -a "$LOG" >&2
  exit 1
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export TZ="${TZ:-America/Sao_Paulo}"

code=0
{
  echo "=============================================================="
  echo "inicio: $(date -Is)  python: $PY  args: ${*:-nenhum}"
  "$PY" -u scripts/05_atualizacao_semanal.py "$@"
  code=$?
  echo "fim:    $(date -Is)  exit=$code"
} >>"$LOG" 2>&1

# 0 = ok (inclui "sem novidade"); 3 = outra execucao em andamento, nao e falha
if [ "$code" -ne 0 ] && [ "$code" -ne 3 ] && [ -n "${ALERTA_WEBHOOK:-}" ]; then
  motivo="$(tail -n 25 "$LOG" | tr '"' "'" | tr '\n' ' ')"
  curl -fsS -m 15 -X POST -H 'Content-Type: application/json' \
    -d "{\"texto\":\"Diesel S-10: job semanal falhou (exit=$code). $motivo\"}" \
    "$ALERTA_WEBHOOK" >>"$LOG" 2>&1 || echo "alerta nao entregue" >>"$LOG"
fi

find "$ROOT/logs" -name 'semanal-*.log' -mtime +60 -delete 2>/dev/null || true

exit "$code"
