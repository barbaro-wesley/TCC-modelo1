"""Job semanal: baixa os dados novos da ANP e retreina o modelo de producao.

E este o script que o cron da VPS chama. Sem semana nova ele sai em segundos,
entao pode ser agendado todo dia sem desperdicio.

    python scripts/05_atualizacao_semanal.py                 # so retreina se houver semana nova
    python scripts/05_atualizacao_semanal.py --forcar        # retreina de qualquer jeito
    python scripts/05_atualizacao_semanal.py --somente-dados # baixa e reconstroi, sem treinar
    python scripts/05_atualizacao_semanal.py --pular-lstm    # dispensa o torch no benchmark

Codigos de saida: 0 ok (inclui "sem novidade"), 1 erro, 2 dados reprovados na
sanidade, 3 outra execucao em andamento.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.weekly import DadosSuspeitos, PipelineOcupado, executar_com_trava, log  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Atualizacao semanal dos dados e do modelo")
    ap.add_argument("--forcar", action="store_true", help="retreina mesmo sem semana nova da ANP")
    ap.add_argument("--somente-dados", action="store_true", help="baixa e reconstroi features, sem treinar")
    ap.add_argument("--pular-lstm", action="store_true", help="pula o benchmark LSTM (nao precisa de torch)")
    args = ap.parse_args()

    try:
        resultado = executar_com_trava(
            forcar=args.forcar,
            somente_dados=args.somente_dados,
            pular_lstm=args.pular_lstm,
        )
    except PipelineOcupado as exc:
        log(f"OCUPADO: {exc}")
        return 3
    except DadosSuspeitos as exc:
        log(f"DADOS REPROVADOS: {exc}")
        log("Os dados processados anteriores foram mantidos. Verifique a planilha da ANP.")
        return 2
    except Exception as exc:  # noqa: BLE001 - o cron precisa do motivo no log
        log(f"FALHA: {type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
        return 1

    log(f"status={resultado['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
