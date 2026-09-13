from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.download import download_all  # noqa: E402
from data.build import save_processed  # noqa: E402


def main():
    paths = download_all()
    print("Arquivos baixados:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    out = save_processed()
    gate = out["gate"]
    print("\nGate Tabela 1 (S10 mensal dez/2012-mai/2020):")
    print("  ok =", gate["ok"])
    for k, v in gate["observed"].items():
        exp = gate["expected"][k]
        print(f"  {k:8s} obs={v:.4f}  art={exp}  diff={gate['diffs'][k]:.4f}" if k != "n"
              else f"  {k:8s} obs={v}  art={exp}  diff={gate['diffs'][k]}")
    print(f"\nMensal completo: {len(out['monthly'])} linhas")
    print(f"Semanal completo: {len(out['weekly'])} linhas")
    print(f"Janela artigo: {len(out['paper'])} linhas")


if __name__ == "__main__":
    main()
