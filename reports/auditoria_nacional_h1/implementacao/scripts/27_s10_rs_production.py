"""Treina o modelo estadual, emite a previsao operacional e registra no ledger.

O que este script produz
------------------------
1. Um artefato imutavel verificado por SHA-256, no mesmo espirito dos demais.
2. A previsao da semana seguinte para o estado, com intervalo calibrado.
3. O **relatorio de base**: quanto custa, em R$/ano, orcar pela media nacional em
   vez da serie do estado.  E o numero que sustenta a conversa comercial, e ele
   nao depende de nenhum modelo — e medicao.
4. Um ledger prospectivo append-only, encadeado por SHA-256, que liquida a
   previsao da semana anterior assim que a ANP publica o valor oficial.

Sobre a evidencia
-----------------
O modelo estadual foi avaliado **apenas em desenvolvimento**.  O holdout do
estado nunca foi lido e continua fechado.  O payload carrega isso explicitamente
em ``evidence.status``: quem consumir a previsao sabe em que terreno esta
pisando, sem precisar ler relatorio nenhum.

Uso::

    python scripts/26_s10_rs_regional.py      # ingere e avalia
    python scripts/27_s10_rs_production.py    # treina, preve e registra
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vs_epl_krls.audit import (  # noqa: E402
    record_forecast,
    settle_pending_forecast,
    verify_audit_ledger,
)
from vs_epl_krls.calibration import backtest_adaptive_interval  # noqa: E402
from vs_epl_krls.gates import interval_report  # noqa: E402
from vs_epl_krls.passthrough import (  # noqa: E402
    PARITY_FEATURES,
    PassThroughConfig,
    PassThroughECM,
    build_parity_panel,
)
from vs_epl_krls.regional import UF_REGION, SpreadForecaster  # noqa: E402

CONTRACT_VERSION = "regional-1.0.0"
SIGNAL_THRESHOLD = 0.01
DEFAULT_MONTHLY_LITERS = 200_000.0
FLEXIBILITY = 0.25
PROSPECTIVE_TARGET = 26
NOMINAL_COVERAGE = 0.80

#: Semanas de walk-forward causal usadas para calibrar o nivel do intervalo.
#: A calibracao usa apenas semanas ja realizadas e publicadas pela ANP, e nao
#: seleciona modelo nenhum — o mesmo tratamento que o bundle nacional da a sua
#: janela movel de residuos.
CALIBRATION_WEEKS = 156

#: Janela do relatorio de base.  Um ano cobre o ciclo de safra do Sul sem
#: diluir o nivel corrente em regimes tributarios antigos.
BASIS_WEEKS = 52


def repo_relative(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(panel: pd.DataFrame) -> str:
    payload = panel[["date", "price"]].to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def national_walk_forward(dates: pd.Series, causal_path: Path) -> pd.DataFrame:
    """Previsao nacional causal para as semanas pedidas."""

    causal = pd.read_csv(causal_path, parse_dates=["date"])
    panel = build_parity_panel(causal)
    wanted = set(pd.to_datetime(dates))
    positions = [index for index, value in enumerate(panel["date"]) if value in wanted]
    if not positions:
        raise RuntimeError("nenhuma semana do estado casa com o painel nacional")
    start, end = max(1, min(positions)), max(positions) + 1
    model = PassThroughECM(config=PassThroughConfig(), feature_names=PARITY_FEATURES)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.walk_forward(panel, start, end, refit_every=1)
    return result[["date", "prediction"]].rename(columns={"prediction": "national"})


def calibrated_band(
    panel: pd.DataFrame, causal_path: Path
) -> tuple[object, dict[str, object]]:
    """Aprende o nivel do intervalo estadual sobre o passado recente.

    O residuo aqui e o do modelo servido inteiro — erro nacional mais erro de
    spread — porque e esse o erro que o cliente enfrenta.
    """

    start = max(1, len(panel) - CALIBRATION_WEEKS)
    window = panel.iloc[start:]
    national = national_walk_forward(window["date"], causal_path)
    aligned = (
        window[["date"]].merge(national, on="date", how="left")["national"].to_numpy(float)
    )
    history = SpreadForecaster(use_anchor=False).walk_forward(
        panel, aligned, start, len(panel)
    )
    usable = history.dropna(subset=["actual", "prediction"])
    if len(usable) < 20:
        return None, {"available": False, "reason": "historico insuficiente para calibrar"}

    result = backtest_adaptive_interval(
        usable["actual"].to_numpy(float),
        usable["prediction"].to_numpy(float),
        nominal_coverage=NOMINAL_COVERAGE,
        gamma=0.02,
        window=CALIBRATION_WEEKS,
        min_residuals=20,
        normalize=False,
    )
    calibrated = interval_report(
        usable["actual"].to_numpy(float),
        result["lower"],
        result["upper"],
        nominal_coverage=NOMINAL_COVERAGE,
    )
    return result["band"], {
        "available": True,
        "calibration_weeks": int(len(usable)),
        "adaptive_conformal": calibrated.as_dict(),
        "state": result["state"].as_dict(),
    }


def basis_report(panel: pd.DataFrame, uf: str) -> dict[str, object]:
    """Quanto custa orcar pela media nacional em vez da serie do estado.

    Devolve numeros **por litro**, para que o consumidor multiplique pelo proprio
    volume sem que o produto precise saber dele.
    """

    spread = panel["spread"].dropna()
    recent = spread.tail(BASIS_WEEKS)
    current = float(spread.iloc[-1])
    mean, deviation = float(spread.mean()), float(spread.std())
    last = panel.dropna(subset=["spread"]).iloc[-1]
    return {
        "uf": uf,
        "as_of": str(pd.Timestamp(last["date"]).date()),
        "state_price_brl_per_liter": float(last["price"]),
        "national_price_brl_per_liter": float(last["national_price"]),
        "current_spread_brl_per_liter": current,
        "relative_difference": current / float(last["national_price"]),
        "window_weeks": BASIS_WEEKS,
        # O erro de base e o desvio *absoluto*: orcar pela media nacional erra
        # para cima ou para baixo, e as duas direcoes custam.
        "mean_absolute_spread_brl_per_liter": float(recent.abs().mean()),
        "spread_z": (current - mean) / deviation if deviation else float("nan"),
        "spread_percentile": float((spread < current).mean()),
        "history_weeks": int(len(spread)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uf", default="RS")
    parser.add_argument("--panel", type=Path, default=None)
    parser.add_argument(
        "--causal", type=Path, default=ROOT / "data" / "processed" / "s10_causal_panel.csv"
    )
    parser.add_argument(
        "--national-forecast",
        type=Path,
        default=ROOT / "reports" / "vs_epl_krls" / "s10_parity" / "latest_forecast.json",
    )
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--monthly-liters", type=float, default=DEFAULT_MONTHLY_LITERS)
    args = parser.parse_args()

    uf = str(args.uf).strip().upper()
    if uf not in UF_REGION:
        raise SystemExit(f"unidade da federacao desconhecida: {uf}")
    panel_path = args.panel or ROOT / "data" / "processed" / f"s10_panel_{uf.lower()}.csv"
    output_dir = args.output_dir or ROOT / "reports" / "vs_epl_krls" / f"s10_{uf.lower()}"
    artifact = args.artifact or ROOT / "artifacts" / f"s10_{uf.lower()}.joblib"
    ledger = args.ledger or output_dir / f"{uf.lower()}_ledger.jsonl"
    if not panel_path.is_file():
        raise SystemExit(f"painel ausente: {panel_path}; rode 26_s10_rs_regional.py antes")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact.parent.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(panel_path, parse_dates=["date"])
    observed = {
        str(pd.Timestamp(date).date()): float(price)
        for date, price in zip(panel["date"], panel["price"])
        if pd.notna(price)
    }
    settled = settle_pending_forecast(ledger, observed, model_name=f"regional_{uf.lower()}")

    national = json.loads(args.national_forecast.read_text(encoding="utf-8"))
    national_forecast = national["forecast"]
    target_date = str(national_forecast["target_date"])
    national_point = float(national_forecast["point"])

    last = panel.dropna(subset=["spread"]).iloc[-1]
    origin_date = pd.Timestamp(last["date"])
    if str((origin_date + pd.Timedelta(days=7)).date()) != target_date:
        raise SystemExit(
            f"a previsao nacional aponta para {target_date}, mas a ultima semana "
            f"observada do {uf} e {origin_date.date()}: reingira o painel estadual"
        )

    model = SpreadForecaster(use_anchor=False).fit(panel)
    next_row = pd.Series(
        {
            "date": origin_date + pd.Timedelta(days=7),
            "spread_lag1": float(last["spread"]),
            "producer_spread_z": float(last.get("producer_spread_z", np.nan)),
        }
    )
    forecast = model.forecast_row(next_row, national_point=national_point)

    band, calibration = calibrated_band(panel, args.causal)
    if band is not None:
        lower, upper = band.interval(forecast.state_point)
        interval_method = "adaptive_conformal"
    else:
        half = 1.2815515655446004 * (model.sigma_ or 0.02)
        lower, upper = forecast.state_point - half, forecast.state_point + half
        interval_method = "gaussian_fallback"

    origin_price = float(last["price"])
    expected_rise = forecast.state_point - origin_price
    recommend = bool(expected_rise > SIGNAL_THRESHOLD)
    weekly_liters = args.monthly_liters * 12.0 / 52.0
    liters = weekly_liters * FLEXIBILITY if recommend else 0.0
    basis = basis_report(panel, uf)

    payload: dict[str, object] = {
        "contract_version": CONTRACT_VERSION,
        "uf": uf,
        "region": UF_REGION[uf],
        "generated_from_panel": repo_relative(panel_path),
        "data_fingerprint": fingerprint(panel),
        "n_train": model.n_train_,
        "spread_model": model.summary(),
        "forecast": {
            "origin_date": str(origin_date.date()),
            "target_date": target_date,
            "origin_price": round(origin_price, 6),
            "national_point": round(national_point, 6),
            "spread_point": round(forecast.spread_point, 6),
            "point": round(forecast.state_point, 6),
            "delta": round(expected_rise, 6),
            "lower": round(float(lower), 6),
            "upper": round(float(upper), 6),
            "nominal_coverage": NOMINAL_COVERAGE,
            "interval_method": interval_method,
            "fallback_used": forecast.fallback_used,
            "reason": forecast.reason,
        },
        "decision": {
            "policy": "antecipar 25% de uma semana quando a alta prevista supera R$ 0,01/L",
            "signal_threshold_brl_per_liter": SIGNAL_THRESHOLD,
            "expected_change_brl_per_liter": round(expected_rise, 6),
            "recommend_prebuy": recommend,
            "liters_if_triggered": round(liters, 1) if recommend else 0.0,
            "expected_saving_brl": round(expected_rise * liters, 2) if recommend else 0.0,
        },
        "basis": basis,
        "interval_calibration": calibration,
        "evidence": {
            # Declarado no payload de proposito: quem consome a previsao precisa
            # saber que ela nunca viu um holdout, sem ter de ler relatorio.
            "status": "development_only",
            "holdout_read": False,
            "prospective_weeks_settled": 0,
            "prospective_target": PROSPECTIVE_TARGET,
            "note": (
                "avaliado apenas em folds de desenvolvimento ate 2024-08-11; "
                "o holdout estadual permanece fechado"
            ),
        },
    }

    joblib.dump(
        {"model": model, "config": asdict(model.config), "payload": payload},
        artifact,
        compress=3,
    )
    payload["artifact_sha256"] = sha256_file(artifact)
    payload["artifact_bytes"] = artifact.stat().st_size

    forecast_record = record_forecast(ledger, payload, target_date=target_date)
    records = verify_audit_ledger(ledger)
    realized = [record for record in records if record["event"] == "realized"]
    payload["evidence"]["prospective_weeks_settled"] = len(realized)
    (output_dir / "latest_forecast.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"{uf}: treino com {model.n_train_} semanas ate {origin_date.date()}")
    print(f"modelo de spread: {json.dumps(model.summary(), ensure_ascii=False)}")
    print()
    print(f"origem  {origin_date.date()}  R$ {origin_price:.4f}/L")
    print(f"alvo    {target_date}")
    print(f"nacional R$ {national_point:.4f}/L  +  spread R$ {forecast.spread_point:+.4f}/L")
    print(f"ponto   R$ {forecast.state_point:.4f}/L  (variacao {expected_rise:+.4f})")
    print(f"P10-P90 R$ {float(lower):.4f} a {float(upper):.4f}/L ({interval_method})")
    if calibration.get("available"):
        adaptive = calibration["adaptive_conformal"]
        print(f"  cobertura calibrada {adaptive['empirical_coverage']:.1%} em "
              f"{calibration['calibration_weeks']} semanas")
    print()
    print(f"decisao: {'ANTECIPAR' if recommend else 'AGUARDAR'} "
          f"({args.monthly_liters:,.0f} L/mes)")
    if recommend:
        print(f"  comprar {liters:,.0f} L uma semana antes; "
              f"economia esperada R$ {expected_rise * liters:,.2f}")

    litres_year = args.monthly_liters * 12.0
    print()
    print(f"base: {uf} R$ {basis['state_price_brl_per_liter']:.4f}/L contra nacional "
          f"R$ {basis['national_price_brl_per_liter']:.4f}/L "
          f"({basis['relative_difference']:+.2%})")
    print(f"  erro de orcamento por usar a serie nacional: "
          f"R$ {basis['mean_absolute_spread_brl_per_liter']:.4f}/L -> "
          f"R$ {basis['mean_absolute_spread_brl_per_liter'] * litres_year:,.0f}/ano")
    print(f"  spread no percentil {basis['spread_percentile']:.1%} de "
          f"{basis['history_weeks']} semanas (z = {basis['spread_z']:+.2f})")

    print()
    if settled is not None:
        scored = settled["payload"]
        print(f"semana liquidada {scored['target_date']}: observado "
              f"R$ {scored['observed_price']:.4f}/L, erro R$ {scored['absolute_error']:.4f}/L, "
              f"persistencia R$ {scored['persistence_absolute_error']:.4f}/L")
    if forecast_record is None:
        print("ledger: previsao identica ja registrada para esta semana-alvo")
    elif forecast_record["event"] == "forecast_revision":
        print("ledger: revisao registrada; o artefato desta semana-alvo mudou")
    print(f"artefato {artifact}")
    print(f"sha256   {payload['artifact_sha256']}")
    print(f"ledger   {ledger}")
    print(f"registros {len(records)}, head {records[-1]['record_hash'][:16]}")
    print(f"contagem prospectiva: {len(realized)}/{PROSPECTIVE_TARGET} semanas liquidadas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
