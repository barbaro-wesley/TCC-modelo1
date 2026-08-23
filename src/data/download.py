from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import requests

from data.anp import URLS

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
TIMEOUT = 120
RETRIES = 3
BACKOFF = 5.0
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "*/*",
}


class DownloadError(RuntimeError):
    """Rede caiu ou o conteudo veio invalido; o arquivo anterior nao foi tocado."""


def _promote(part: Path, dest: Path) -> None:
    """Substitui dest por part guardando a versao anterior em .prev."""
    if dest.exists() and dest.stat().st_size > 0:
        shutil.copy2(dest, dest.with_suffix(dest.suffix + ".prev"))
    part.replace(dest)


def download_file(
    url: str,
    dest: Path,
    force: bool = False,
    validate: Optional[Callable[[Path], None]] = None,
    retries: int = RETRIES,
) -> Path:
    """Baixa para .part, valida e so entao substitui o destino.

    Sem validate o comportamento e o antigo: qualquer resposta 200 vale. Com
    validate, uma pagina de erro ou de captcha nunca sobrescreve dado bom.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            r.raise_for_status()
            if not r.content:
                raise DownloadError("resposta vazia")
            part.write_bytes(r.content)
            if validate is not None:
                validate(part)
            _promote(part, dest)
            return dest
        except Exception as exc:  # rede, HTTP ou conteudo invalido
            last_exc = exc
            part.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(BACKOFF * attempt)
    raise DownloadError(f"{url}: {last_exc}") from last_exc


def validate_anp_xlsx(path: Path, min_linhas: int) -> None:
    """O limiar e por planilha: a de 2001-2012 tem 1 linha de S10 (o produto so
    aparece em 2013), a mensal desde 2013 tem ~160 e a semanal ~700."""
    from data.anp import filter_s10, read_anp_sheet

    df = filter_s10(read_anp_sheet(path))
    if len(df) < min_linhas:
        raise DownloadError(f"planilha ANP com apenas {len(df)} linhas de S10 (esperado >= {min_linhas})")


def validate_ipeadata(path: Path, min_pontos: int = 500) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    valores = payload.get("value") if isinstance(payload, dict) else payload
    n = len(valores) if isinstance(valores, list) else 0
    if n < min_pontos:
        raise DownloadError(f"serie IPEADATA com {n} pontos")


def validate_stooq_csv(path: Path) -> None:
    head = path.read_text(encoding="utf-8", errors="replace")[:400].lower()
    if "<html" in head or "<!doctype" in head:
        raise DownloadError("Stooq devolveu HTML (bot challenge), nao CSV")
    if "date" not in head or "close" not in head:
        raise DownloadError("CSV do Stooq sem colunas date/close")


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


def _baixar_serie_json(
    dest: Path,
    fetch: Callable[[], object],
    validate: Callable[[Path], None],
    force: bool,
    obrigatorio: bool,
    fontes: dict,
    chave: str,
) -> Optional[Path]:
    """Serie de API JSON: escreve em .part, valida, promove.

    Se falhar e ja houver copia anterior, mantem a anterior e registra o aviso
    em vez de derrubar a execucao semanal inteira.
    """
    if dest.exists() and dest.stat().st_size > 0 and not force:
        fontes[chave] = {"status": "cache", "arquivo": dest.name}
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    last_exc: Optional[Exception] = None
    for attempt in range(1, RETRIES + 1):
        try:
            part.write_text(json.dumps(fetch()), encoding="utf-8")
            validate(part)
            _promote(part, dest)
            fontes[chave] = {"status": "atualizado", "arquivo": dest.name}
            return dest
        except Exception as exc:
            last_exc = exc
            part.unlink(missing_ok=True)
            if attempt < RETRIES:
                time.sleep(BACKOFF * attempt)
    if dest.exists() and dest.stat().st_size > 0:
        fontes[chave] = {"status": "mantido_anterior", "erro": str(last_exc), "arquivo": dest.name}
        return dest
    fontes[chave] = {"status": "falhou", "erro": str(last_exc)}
    if obrigatorio:
        raise DownloadError(f"{chave}: {last_exc}") from last_exc
    return None


def download_all(force: bool = False) -> dict:
    """Baixa tudo. force=True e o modo do cron semanal (ignora o cache em disco).

    Escreve data/raw/download_report.json com o que foi atualizado, mantido ou
    perdido, para o orquestrador semanal decidir se pode seguir.
    """
    RAW.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    relatorio: dict = {"executado_em": stamp, "force": force, "fontes": {}}
    fontes = relatorio["fontes"]
    paths: dict = {}

    def _salvar_relatorio() -> None:
        (RAW / "download_report.json").write_text(
            json.dumps(relatorio, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # O minimo e uma margem sobre o tamanho atual de cada planilha: pega
    # arquivo truncado ou pagina de erro sem reprovar uma coleta legitima.
    planilhas = [
        ("mensal_2013", URLS["mensal_2013"], RAW / "anp_mensal_desde_2013.xlsx", 120),
        ("mensal_2001", URLS["mensal_2001"], RAW / "anp_mensal_2001_2012.xlsx", 1),
        ("semanal_2013", URLS["semanal_2013"], RAW / "anp_semanal_desde_2013.xlsx", 500),
    ]
    for chave, url, dest, minimo in planilhas:
        try:
            paths[chave] = download_file(
                url, dest, force, validate=lambda p, m=minimo: validate_anp_xlsx(p, m)
            )
            fontes[chave] = {"status": "atualizado" if force else "ok", "arquivo": dest.name}
        except Exception as exc:
            fontes[chave] = {"status": "falhou", "erro": str(exc)}
            _salvar_relatorio()
            raise

    brent = _baixar_serie_json(
        RAW / "ipeadata_brent.json",
        lambda: fetch_ipeadata("EIA366_PBRENT366"),
        validate_ipeadata,
        force,
        obrigatorio=True,
        fontes=fontes,
        chave="brent",
    )
    fx = _baixar_serie_json(
        RAW / "ipeadata_usdbrl.json",
        lambda: fetch_ipeadata("GM366_ERC366"),
        validate_ipeadata,
        force,
        obrigatorio=True,
        fontes=fontes,
        chave="fx",
    )
    if brent is not None:
        paths["brent"] = brent
    if fx is not None:
        paths["fx"] = fx

    try:
        # ULSD e opcional: as features ulsd_* saem do painel quando faltam.
        # retries=1: o Stooq responde com desafio de bot de forma consistente,
        # insistir so gastaria backoff em toda execucao do cron.
        ulsd = download_file(
            "https://stooq.com/q/d/l/?s=ho.f&i=d",
            RAW / "stooq_ulsd.csv",
            force,
            validate=validate_stooq_csv,
            retries=1,
        )
        paths["ulsd"] = ulsd
        fontes["ulsd"] = {"status": "atualizado", "arquivo": ulsd.name}
    except Exception as exc:
        (RAW / "stooq_ulsd.error").write_text(str(exc), encoding="utf-8")
        fontes["ulsd"] = {"status": "indisponivel", "erro": str(exc)}

    (RAW / "download_stamp.txt").write_text(stamp, encoding="utf-8")
    _salvar_relatorio()
    return paths
