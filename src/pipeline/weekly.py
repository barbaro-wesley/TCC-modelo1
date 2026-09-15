"""Job independente: treina em arquivos locais e publica resultados no PostgreSQL.

CSVs, figuras e diagnosticos sao artefatos de pesquisa do treinamento.
A API consulta somente publicacoes completas no banco.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
from filelock import FileLock, Timeout

from eval.temporal import TEMPORAL_PROTOCOL

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
RES = ROOT / "results"
LOCK = RES / ".pipeline.lock"


class PipelineOcupado(RuntimeError):
    """Outra execucao esta em andamento."""


class DadosSuspeitos(RuntimeError):
    """A planilha nova nao passou nas checagens; o processado antigo foi mantido."""


def agora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[{agora()}] {msg}", flush=True)


@contextmanager
def trava():
    RES.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(LOCK), timeout=0)
    try:
        lock.acquire()
    except Timeout as exc:
        raise PipelineOcupado("outro treinamento usa este diretorio") from exc
    try:
        yield
    finally:
        lock.release()


def rodar_script(nome: str, env_extra: Optional[dict] = None) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(ROOT / "src"), env.get("PYTHONPATH", "")))
    env.setdefault("MPLBACKEND", "Agg")
    if env_extra:
        env.update(env_extra)
    log(f"executando {nome}")
    proc = subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / nome)], cwd=str(ROOT), env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"{nome} terminou com codigo {proc.returncode}")


def checar_sanidade(semanal: pd.DataFrame, estado: dict) -> List[str]:
    """Barreiras contra planilha truncada, datas absurdas ou salto de preco irreal."""
    problemas: List[str] = []
    n_anterior = estado.get("n_linhas_semanal")
    if n_anterior and len(semanal) < n_anterior:
        problemas.append(f"planilha encolheu: {len(semanal)} linhas contra {n_anterior} da ultima vez")
    if len(semanal) < 300:
        problemas.append(f"serie semanal com apenas {len(semanal)} linhas")

    ultima = pd.Timestamp(semanal["data"].max())
    if ultima > pd.Timestamp.today().normalize() + pd.Timedelta(days=10):
        problemas.append(f"ultima semana no futuro: {ultima.date()}")

    precos = pd.to_numeric(semanal["revenda"], errors="coerce")
    if precos.tail(12).isna().any():
        problemas.append("ha precos vazios nas ultimas 12 semanas")
    preco_novo = float(precos.iloc[-1])
    preco_ant = estado.get("preco_ultima_semana")
    if preco_ant:
        razao = preco_novo / float(preco_ant)
        if not 0.5 <= razao <= 2.0:
            problemas.append(
                f"salto de preco implausivel: {preco_ant} -> {preco_novo} (x{razao:.2f})"
            )
    if not 1.0 <= preco_novo <= 30.0:
        problemas.append(f"preco fora da faixa esperada: {preco_novo}")
    return problemas


def executar(forcar=False, somente_dados=False, pular_lstm=False, *, publisher=None):
    from data.build import load_weekly_s10, save_processed
    from data.download import download_all
    from pipeline.forecast import montar_previsao

    owned = publisher is None
    if owned:
        from training.database import publisher_from_environment

        publisher = publisher_from_environment()
    run_id = None
    sources = {}
    started = time.monotonic()
    try:
        run_id = publisher.start(
            code_version=os.environ.get("S10_CODE_VERSION", "unknown"),
            protocol=TEMPORAL_PROTOCOL,
            config={"forcar": forcar, "somente_dados": somente_dados, "pular_lstm": pular_lstm},
        )
        estado = publisher.state()
        download_all(force=True)
        report = RAW / "download_report.json"
        if report.exists():
            sources = json.loads(report.read_text(encoding="utf-8"))
        semanal = load_weekly_s10().sort_values("data").reset_index(drop=True)
        problemas = checar_sanidade(semanal, estado)
        if problemas:
            raise DadosSuspeitos("; ".join(problemas))
        ultima = str(pd.Timestamp(semanal["data"].max()).date())
        if (not forcar and not somente_dados
                and estado.get("ultima_semana_processada") == ultima
                and estado.get("temporal_protocol") == TEMPORAL_PROTOCOL):
            publisher.finish(run_id, "no_change", sources=sources)
            return {"status": "sem_novidade", "run_id": run_id}
        save_processed()
        if somente_dados:
            publisher.finish(run_id, "data_only", sources=sources)
            return {"status": "dados_atualizados", "run_id": run_id}
        rodar_script("03_semanal.py", {"SKIP_LSTM": "1"} if pular_lstm else None)
        previsao = montar_previsao()
        serie = [
            {"data": str(pd.Timestamp(d).date()), "revenda": float(v)}
            for d, v in zip(semanal["data"], pd.to_numeric(semanal["revenda"], errors="coerce"))
            if pd.notna(v)
        ]
        metricas = pd.read_csv(RES / "semanal_benchmarks.csv").to_dict(orient="records")
        status = publisher.publish(run_id, previsao, serie, metricas, sources=sources)
        return {
            "status": "atualizado" if status == "published" else "substituido",
            "run_id": run_id, "modelo": previsao["modelo"],
            "semana_prevista": previsao["semana_prevista"],
            "duracao_segundos": round(time.monotonic() - started, 2),
        }
    except Exception as exc:
        if run_id is not None:
            try:
                publisher.finish(run_id, "failed", sources=sources, error_code=type(exc).__name__)
            except Exception:
                log("Nao foi possivel registrar a falha no banco; consulte o log do job.")
        raise
    finally:
        if owned:
            publisher.engine.dispose()


def executar_com_trava(**kwargs):
    with trava():
        return executar(**kwargs)
