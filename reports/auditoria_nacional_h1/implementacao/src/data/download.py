from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import requests

from data.anp import URLS

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
TIMEOUT = 120
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "*/*",
}


def download_file(url: str, dest: Path, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def fetch_ipeadata(sercodigo: str) -> list:
    url = f"http://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='{sercodigo}')"
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["value"]


def fetch_bcb(serie: int, start: str = "01/01/2012") -> list:
    url = (
        f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{serie}/dados"
        f"?formato=json&dataInicial={start}"
    )
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def fetch_stooq(symbol: str) -> str:
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return r.text


def download_all(force: bool = False) -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    paths = {
        "mensal_2013": download_file(URLS["mensal_2013"], RAW / "anp_mensal_desde_2013.xlsx", force),
        "mensal_2001": download_file(URLS["mensal_2001"], RAW / "anp_mensal_2001_2012.xlsx", force),
        "semanal_2013": download_file(URLS["semanal_2013"], RAW / "anp_semanal_desde_2013.xlsx", force),
    }
    import json

    brent = fetch_ipeadata("EIA366_PBRENT366")
    (RAW / "ipeadata_brent.json").write_text(json.dumps(brent), encoding="utf-8")
    fx = fetch_ipeadata("GM366_ERC366")
    (RAW / "ipeadata_usdbrl.json").write_text(json.dumps(fx), encoding="utf-8")
    try:
        ulsd = fetch_stooq("ho.f")
        (RAW / "stooq_ulsd.csv").write_text(ulsd, encoding="utf-8")
        paths["ulsd"] = RAW / "stooq_ulsd.csv"
    except Exception as exc:
        (RAW / "stooq_ulsd.error").write_text(str(exc), encoding="utf-8")
    (RAW / "download_stamp.txt").write_text(stamp, encoding="utf-8")
    paths["brent"] = RAW / "ipeadata_brent.json"
    paths["fx"] = RAW / "ipeadata_usdbrl.json"
    return paths
