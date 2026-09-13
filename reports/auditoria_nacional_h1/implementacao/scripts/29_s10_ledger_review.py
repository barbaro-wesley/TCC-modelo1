"""Revisa os ledgers prospectivos e falha quando algo exige atencao.

Feito para rodar sozinho — cron semanal, passo de CI, o que for.  Sai com codigo
diferente de zero quando ha alerta critico, para que o agendador reclame em vez
de escrever num log que ninguem le.

Uso::

    python scripts/29_s10_ledger_review.py
    python scripts/29_s10_ledger_review.py --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vs_epl_krls.monitoring import review_ledgers  # noqa: E402

REPORTS = ROOT / "reports" / "vs_epl_krls"

DEFAULT_LEDGERS = {
    "paridade": REPORTS / "s10_parity" / "parity_ledger.jsonl",
    "regional_rs": REPORTS / "s10_rs" / "rs_ledger.jsonl",
}

_ICON = {"critical": "X ", "warning": "! ", "info": "i ", "ok": "OK"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="saida legivel por maquina")
    parser.add_argument(
        "--ledger",
        action="append",
        default=None,
        metavar="NOME=CAMINHO",
        help="ledger adicional; repita para varios",
    )
    args = parser.parse_args()

    ledgers = dict(DEFAULT_LEDGERS)
    for entry in args.ledger or []:
        name, _, path = str(entry).partition("=")
        if not path:
            raise SystemExit(f"formato esperado NOME=CAMINHO, recebi {entry!r}")
        ledgers[name] = Path(path)

    statuses = review_ledgers(ledgers)
    if args.json:
        print(
            json.dumps(
                {name: status.as_dict() for name, status in statuses.items()},
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        if not statuses:
            print("nenhum ledger encontrado")
        for name, status in statuses.items():
            print(f"{_ICON[status.worst_level]} {name}")
            print(f"     {status.settled}/{status.target} semanas liquidadas, "
                  f"{status.forecasts} previsao(oes) registrada(s)")
            if status.pending_target_date:
                print(f"     pendente: {status.pending_target_date} "
                      f"({status.weeks_pending} semana(s))")
            if status.coverage is not None:
                print(f"     cobertura {status.coverage:.1%}")
            if status.mae is not None:
                comparison = (
                    f" contra {status.persistence_mae:.4f} da persistencia"
                    if status.persistence_mae is not None
                    else ""
                )
                print(f"     MAE {status.mae:.4f}{comparison}")
            for alert in status.alerts:
                print(f"     [{alert.level}] {alert.code}: {alert.message}")

    critical = [
        alert
        for status in statuses.values()
        for alert in status.alerts
        if alert.level == "critical"
    ]
    if critical:
        print(f"{chr(10)}{len(critical)} alerta(s) critico(s)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
