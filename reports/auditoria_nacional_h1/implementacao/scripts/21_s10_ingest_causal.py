"""Baixa e consolida as fontes causais do S10 num painel versionado.

Executa a ingestao completa (ULSD, Brent, dolar, preco de produtor da ANP),
alinha tudo ao indice semanal da ANP e grava o painel com manifesto de
proveniencia.

Uso::

    python scripts/21_s10_ingest_causal.py
    python scripts/21_s10_ingest_causal.py --offline   # usa o painel ja gravado
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vs_epl_krls.causal_ingest import build_causal_panel  # noqa: E402

DEFAULT_RETAIL = ROOT / "data" / "processed" / "semanal_s10_features.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "s10_causal_panel.csv"
DEFAULT_MANIFEST = ROOT / "data" / "processed" / "s10_causal_panel_manifest.json"


def load_retail(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["data"])
    return frame.rename(columns={"data": "date", "revenda": "price"})[["date", "price"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retail", type=Path, default=DEFAULT_RETAIL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument(
        "--offline", action="store_true", help="apenas inspeciona o painel ja gravado"
    )
    args = parser.parse_args()

    if args.offline:
        if not args.output.is_file():
            print(f"painel ausente: {args.output}")
            return 1
        panel = pd.read_csv(args.output, parse_dates=["date"])
        print(f"painel offline: {len(panel)} semanas, "
              f"{panel['date'].min().date()} a {panel['date'].max().date()}")
        return 0

    retail = load_retail(args.retail)
    print(f"revenda: {len(retail)} semanas, {retail['date'].min().date()} a "
          f"{retail['date'].max().date()}")
    print("baixando fontes causais...")
    panel, manifest = build_causal_panel(retail, raw_dir=args.raw_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.output, index=False)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\npainel: {manifest['n_weeks']} semanas, "
          f"{manifest['coverage']['start']} a {manifest['coverage']['end']}")
    print("\nfontes:")
    for source in manifest["sources"]:
        print(f"  {source['name']:14s} {source['rows']:6d} linhas  "
              f"{source['coverage_start']} .. {source['coverage_end']}  "
              f"sha256 {source['sha256'][:12]}…")
    print("\ncobertura por coluna:")
    for column, count in manifest["coverage_by_column"].items():
        print(f"  {column:16s} {count}/{manifest['n_weeks']}")
    print(f"\ngravado: {args.output}")
    print(f"manifesto: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
