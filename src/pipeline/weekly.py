"""Atualizacao semanal ponta a ponta, feita para rodar sozinha no cron da VPS.

Fluxo: trava -> download validado -> ha semana nova da ANP? -> checagens de
sanidade -> rebuild das features -> walk-forward -> previsao do modelo vencedor
-> relatorios -> historico -> payloads JSON para a API.

Sem semana nova o job sai em segundos sem tocar em nada. Isso permite agendar
diariamente sem saber o dia exato em que a ANP publica.
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

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
REP = ROOT / "reports"
API = RES / "api"

ESTADO = RES / "pipeline_state.json"
LOCK = RES / ".pipeline.lock"
HISTORICO = RES / "historico_previsoes.json"
PREVISAO = RES / "previsao_proxima_semana.json"

LOCK_MAX_HORAS = 6
MARCADOR_INICIO = "<!-- AUTO:PREVISAO:INICIO -->"
MARCADOR_FIM = "<!-- AUTO:PREVISAO:FIM -->"


class PipelineOcupado(RuntimeError):
    """Outra execucao esta em andamento."""


class DadosSuspeitos(RuntimeError):
    """A planilha nova nao passou nas checagens; o processado antigo foi mantido."""


def agora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[{agora()}] {msg}", flush=True)


def ler_json(path: Path, padrao):
    if not path.exists():
        return padrao
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return padrao


def escrever_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


@contextmanager
def trava():
    RES.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        info = ler_json(LOCK, {})
        inicio = info.get("inicio")
        idade_h = None
        if inicio:
            try:
                idade_h = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(inicio)).total_seconds() / 3600
            except Exception:
                idade_h = None
        if idade_h is not None and idade_h < LOCK_MAX_HORAS:
            raise PipelineOcupado(
                f"execucao iniciada em {inicio} (pid {info.get('pid')}) ainda ativa ha {idade_h:.1f}h"
            )
        log(f"AVISO: lock antigo ({inicio}) ignorado")
    LOCK.write_text(json.dumps({"pid": os.getpid(), "inicio": agora()}), encoding="utf-8")
    try:
        yield
    finally:
        LOCK.unlink(missing_ok=True)


def rodar_script(nome: str, env_extra: Optional[dict] = None) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
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


def atualizar_historico(previsao: dict, semanal: pd.DataFrame) -> list:
    """Guarda a previsao da semana e preenche o realizado das semanas passadas."""
    registros = ler_json(HISTORICO, [])
    if not isinstance(registros, list):
        registros = []
    registro = {
        "semana_prevista": previsao["semana_prevista"],
        "semana_referencia": previsao["ultima_semana_observada"],
        "modelo": previsao["modelo"],
        "previsao": previsao["previsao_pontual"],
        "p10": previsao["p10"],
        "p90": previsao["p90"],
        "preco_referencia": previsao["preco_observado_ultima_semana"],
        "gerado_em": agora(),
        "preco_realizado": None,
        "erro": None,
    }
    por_semana = {r.get("semana_prevista"): r for r in registros if isinstance(r, dict)}
    anterior = por_semana.get(registro["semana_prevista"])
    if anterior:
        # mesma semana-alvo reprocessada: preserva quando ela foi prevista pela 1a vez
        registro["gerado_em"] = anterior.get("gerado_em", registro["gerado_em"])
    por_semana[registro["semana_prevista"]] = registro

    realizados = {
        str(pd.Timestamp(d).date()): float(v)
        for d, v in zip(semanal["data"], pd.to_numeric(semanal["revenda"], errors="coerce"))
        if pd.notna(v)
    }
    for semana, reg in por_semana.items():
        real = realizados.get(semana)
        if real is not None:
            reg["preco_realizado"] = real
            if reg.get("previsao") is not None:
                reg["erro"] = round(real - float(reg["previsao"]), 6)
    saida = [por_semana[k] for k in sorted(por_semana)]
    escrever_json(HISTORICO, saida)
    return saida


def escrever_api(previsao: dict, historico: list, semanal: pd.DataFrame, status: dict) -> None:
    """Contrato de leitura da API em Go: tres arquivos estaveis em results/api/."""
    API.mkdir(parents=True, exist_ok=True)
    escrever_json(API / "status.json", status)
    escrever_json(API / "previsao.json", previsao)
    serie = [
        {"data": str(pd.Timestamp(d).date()), "revenda": float(v)}
        for d, v in zip(semanal["data"], pd.to_numeric(semanal["revenda"], errors="coerce"))
        if pd.notna(v)
    ]
    escrever_json(
        API / "historico.json",
        {
            "gerado_em": agora(),
            "unidade": "R$/L",
            "descricao": "Preco medio nacional de revenda do Diesel B S-10 (ANP, semanal)",
            "serie": serie,
            "previsoes": historico,
        },
    )


def atualizar_readme(previsao: dict) -> bool:
    readme = ROOT / "README.md"
    if not readme.exists():
        return False
    texto = readme.read_text(encoding="utf-8")
    if MARCADOR_INICIO not in texto or MARCADOR_FIM not in texto:
        log("AVISO: README sem marcadores AUTO:PREVISAO; secao nao atualizada")
        return False

    def fmt(v):
        return f"{float(v):.2f}".replace(".", ",") if v is not None else "—"

    probs = previsao.get("probabilidades") or {}

    def pct(k):
        v = probs.get(k)
        return f"{100 * float(v):.0f}%" if v is not None else "—"

    linhas = [
        MARCADOR_INICIO,
        "",
        f"Última semana ANP: **{previsao['ultima_semana_observada']}**, "
        f"preço **R$ {fmt(previsao['preco_observado_ultima_semana'])}/L**.",
        "",
        f"Previsão para a semana de **{previsao['semana_prevista']}**, "
        f"modelo **{previsao['modelo']}** (menor RMSE walk-forward em h=1):",
        "",
        "| | R$/L |",
        "| --- | --- |",
        f"| {previsao['modelo']} (produção) | **{fmt(previsao['previsao_pontual'])}** "
        f"({fmt(previsao['p10'])} – {fmt(previsao['p90'])}) |",
    ]
    for nome, valor in (previsao.get("previsoes_por_modelo") or {}).items():
        if nome != previsao["modelo"] and valor is not None:
            linhas.append(f"| {nome} | {fmt(valor)} |")
    linhas += [
        "",
        f"Prob. alta / estável / queda (±0,02): {pct('p_alta')} / {pct('p_estavel')} / {pct('p_queda')}.",
        "",
        f"Atualizado automaticamente em {agora()}. Arquivos: `results/previsao_proxima_semana.json`, `results/api/`.",
        "",
        MARCADOR_FIM,
    ]
    inicio = texto.index(MARCADOR_INICIO)
    fim = texto.index(MARCADOR_FIM) + len(MARCADOR_FIM)
    readme.write_text(texto[:inicio] + "\n".join(linhas) + texto[fim:], encoding="utf-8")
    return True


def executar(forcar: bool = False, somente_dados: bool = False, pular_lstm: bool = False) -> dict:
    t0 = time.time()
    estado = ler_json(ESTADO, {})
    resultado = {"iniciado_em": agora(), "status": "erro", "forcado": forcar}

    from data.build import load_weekly_s10, save_processed
    from data.download import download_all

    log("1/7 baixando dados brutos (ANP, Brent, cambio, ULSD)")
    download_all(force=True)
    relatorio_download = ler_json(RAW / "download_report.json", {})
    for fonte, info in (relatorio_download.get("fontes") or {}).items():
        log(f"    {fonte}: {info.get('status')}" + (f" | {info.get('erro')}" if info.get("erro") else ""))

    log("2/7 lendo a planilha semanal da ANP")
    semanal = load_weekly_s10()
    ultima_semana = str(pd.Timestamp(semanal["data"].max()).date())
    processada = estado.get("ultima_semana_processada")
    log(f"    ultima semana na ANP: {ultima_semana} | ja processada: {processada or 'nenhuma'}")

    resultado.update(
        {
            "ultima_semana_anp": ultima_semana,
            "ultima_semana_processada_antes": processada,
            "fontes": relatorio_download.get("fontes", {}),
        }
    )

    if processada == ultima_semana and not forcar:
        log("sem semana nova: nada a treinar")
        resultado["status"] = "sem_novidade"
        resultado["duracao_s"] = round(time.time() - t0, 1)
        estado.update(
            {
                "ultima_verificacao": agora(),
                "ultimo_status": "sem_novidade",
                "ultima_semana_anp": ultima_semana,
            }
        )
        escrever_json(ESTADO, estado)
        _atualizar_status_api(estado, resultado)
        return resultado

    log("3/7 checagens de sanidade da serie nova")
    problemas = checar_sanidade(semanal, estado)
    if problemas:
        for p in problemas:
            log(f"    REPROVADO: {p}")
        raise DadosSuspeitos("; ".join(problemas))
    log("    ok")

    log("4/7 reconstruindo features processadas")
    saida = save_processed()
    log(f"    semanal={len(saida['weekly'])} linhas | mensal={len(saida['monthly'])} linhas")

    if somente_dados:
        log("modo --somente-dados: treino nao executado")
        resultado["status"] = "dados_atualizados"
        resultado["duracao_s"] = round(time.time() - t0, 1)
        estado.update(
            {
                "ultima_verificacao": agora(),
                "ultimo_status": "dados_atualizados",
                "ultima_semana_anp": ultima_semana,
                "n_linhas_semanal": int(len(semanal)),
                "preco_ultima_semana": float(pd.to_numeric(semanal["revenda"]).iloc[-1]),
            }
        )
        escrever_json(ESTADO, estado)
        _atualizar_status_api(estado, resultado)
        return resultado

    log("5/7 retreinando e reavaliando (walk-forward, h = 1, 2, 4)")
    rodar_script("03_semanal.py", {"SKIP_LSTM": "1"} if pular_lstm else None)

    log("6/7 gerando a previsao do modelo vencedor")
    from pipeline.forecast import montar_previsao

    previsao = montar_previsao(horizon=1)
    log(
        f"    vencedor: {previsao['modelo']} (RMSE wf {previsao['rmse_walkforward']:.5f}) "
        f"-> {previsao['previsao_pontual']:.4f} R$/L para {previsao['semana_prevista']}"
    )
    for descartado in previsao.get("modelos_descartados") or []:
        log(f"    descartado antes do vencedor: {descartado}")
    escrever_json(PREVISAO, previsao)
    rodar_script("04_producao.py")

    log("7/7 historico, payloads da API e README")
    historico = atualizar_historico(previsao, semanal)
    estado.update(
        {
            "ultima_verificacao": agora(),
            "ultima_execucao_ok": agora(),
            "ultimo_status": "atualizado",
            "ultima_semana_processada": ultima_semana,
            "ultima_semana_anp": ultima_semana,
            "n_linhas_semanal": int(len(semanal)),
            "preco_ultima_semana": float(pd.to_numeric(semanal["revenda"]).iloc[-1]),
            "modelo_producao": previsao["modelo"],
        }
    )
    escrever_json(ESTADO, estado)
    resultado.update(
        {
            "status": "atualizado",
            "modelo": previsao["modelo"],
            "previsao": previsao["previsao_pontual"],
            "semana_prevista": previsao["semana_prevista"],
            "duracao_s": round(time.time() - t0, 1),
        }
    )
    escrever_api(previsao, historico, semanal, _status_api(estado, resultado))
    atualizar_readme(previsao)
    log(f"concluido em {resultado['duracao_s']}s")
    return resultado


def _status_api(estado: dict, resultado: dict) -> dict:
    return {
        "gerado_em": agora(),
        "status": resultado.get("status"),
        "ultima_execucao": resultado.get("iniciado_em"),
        "ultima_execucao_ok": estado.get("ultima_execucao_ok"),
        "ultima_verificacao": estado.get("ultima_verificacao"),
        "ultima_semana_anp": resultado.get("ultima_semana_anp") or estado.get("ultima_semana_anp"),
        "ultima_semana_processada": estado.get("ultima_semana_processada"),
        "modelo_producao": estado.get("modelo_producao"),
        "duracao_s": resultado.get("duracao_s"),
        "erro": resultado.get("erro"),
        "fontes": resultado.get("fontes", {}),
    }


def _atualizar_status_api(estado: dict, resultado: dict) -> None:
    """Mantem status.json fresco mesmo quando nao houve retreino."""
    API.mkdir(parents=True, exist_ok=True)
    escrever_json(API / "status.json", _status_api(estado, resultado))


def executar_com_trava(**kwargs) -> dict:
    with trava():
        try:
            return executar(**kwargs)
        except Exception as exc:
            estado = ler_json(ESTADO, {})
            estado.update({"ultima_verificacao": agora(), "ultimo_status": "erro", "ultimo_erro": str(exc)})
            escrever_json(ESTADO, estado)
            _atualizar_status_api(estado, {"status": "erro", "erro": str(exc), "iniciado_em": agora()})
            raise
