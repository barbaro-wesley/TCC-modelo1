from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.classical import arima_forecast, arimax_forecast, ma_predict, naive_predict  # noqa: E402
from benchmarks.gbm import fit_lgbm, fit_xgb  # noqa: E402
from benchmarks.lstm import fit_lstm, predict_lstm  # noqa: E402
from data.build import leak_check  # noqa: E402
from eval.drift import page_hinkley, psi, rolling_rmse  # noqa: E402
from eval.importance import permutation_importance  # noqa: E402
from eval.intervals import conformal_p10_p90, direction_probs  # noqa: E402
from eval.metrics import coverage, diebold_mariano, pinball, summarize  # noqa: E402
from vsepl_krls.model import VSePLKRLS, VSePLKRLSConfig  # noqa: E402

warnings.filterwarnings("ignore")

PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
FIG = ROOT / "reports" / "figures"
REP = ROOT / "reports"

FEATURE_COLS = [
    "revenda_l1", "revenda_l2", "revenda_l4", "revenda_l8", "revenda_l12",
    "revenda_ma4", "revenda_ma8", "revenda_ma12",
    "vol4", "vol12",
    "brent_l1", "brent_l4", "brent_brl_l1", "brent_brl_l4",
    "usdbrl_l1", "usdbrl_l4",
    "ulsd_l1", "ulsd_l4",
    "petrobras_reajuste_l1", "paridade_z_l1",
]

ARIMAX_COLS = ["brent_l1", "usdbrl_l1"]


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.5f}" if abs(v) < 100 else f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_panel(horizon: int) -> pd.DataFrame:
    df = pd.read_csv(PROC / "semanal_s10_features.csv", parse_dates=["data"])
    df = df.sort_values("data").reset_index(drop=True)
    df["y"] = df["revenda"].shift(-horizon)
    df["y_prev"] = df["revenda"]
    future_date = df["data"].shift(-horizon)
    gap_mask = (future_date >= pd.Timestamp("2020-08-18")) & (future_date <= pd.Timestamp("2020-10-17"))
    df = df.loc[~gap_mask].copy()
    feat_cols = [c for c in FEATURE_COLS if c in df.columns and float(df[c].notna().mean()) > 0.8]
    df[feat_cols] = df[feat_cols].ffill()
    keep = feat_cols + ["y", "y_prev", "revenda", "data"]
    out = df[keep].dropna(subset=feat_cols + ["y"]).reset_index(drop=True)
    out.attrs["feat_cols"] = feat_cols
    return out


def minmax_train(Xtr, Xte):
    lo = np.nanmin(Xtr, axis=0)
    hi = np.nanmax(Xtr, axis=0)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    return (Xtr - lo) / span, (Xte - lo) / span, lo, span


def scale_frozen(X, n_min):
    lo = np.nanmin(X[:n_min], axis=0)
    hi = np.nanmax(X[:n_min], axis=0)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    Xs = np.clip((X - lo) / span, -0.25, 1.25)
    return Xs, lo, span


def run_vsepl(X, y, n_min, horizon=1):
    Xs, lo, span = scale_frozen(X, n_min)
    n = len(y)
    yhat = np.full(n, np.nan)
    model = VSePLKRLS(VSePLKRLSConfig(d_max=15, alpha=0.01, beta0=0.18, max_rules=8, threshold_convention="texto"))
    for t in range(n):
        if t >= horizon:
            model.update(Xs[t - horizon], float(y[t - horizon]))
        if t >= n_min:
            yhat[t] = model.predict_one(Xs[t]) if model.n_rules else np.nan
    return yhat, model, (lo, span)


def run_naive(price, n_min):
    yhat = np.full(len(price), np.nan)
    yhat[n_min:] = price[n_min:]
    return yhat


def run_ma(price, n_min, window=4):
    yhat = np.full(len(price), np.nan)
    for t in range(n_min, len(price)):
        yhat[t] = ma_predict(price[: t + 1], window=window, n=1)[0]
    return yhat


def run_arima(price, n_min, horizon, refit_every=12):
    yhat = np.full(len(price), np.nan)
    model = None
    last = -10**9
    for t in range(n_min, len(price)):
        if t - last >= refit_every or model is None:
            fc, model = arima_forecast(price[: t + 1], steps=horizon, model=None)
            last = t
        else:
            fc, model = arima_forecast(price[: t + 1], steps=horizon, model=model)
        yhat[t] = fc[-1]
    return yhat


def run_arimax(price, Xex, n_min, horizon, refit_every=12):
    yhat = np.full(len(price), np.nan)
    model = None
    last = -10**9
    for t in range(n_min, len(price)):
        end = t + 1
        x_hist = Xex[:end]
        x_fut = np.repeat(Xex[t : t + 1], horizon, axis=0)
        if t - last >= refit_every or model is None:
            fc, model = arimax_forecast(price[:end], x_hist, x_fut, steps=horizon, model=None)
            last = t
        else:
            fc, model = arimax_forecast(price[:end], x_hist, x_fut, steps=horizon, model=model)
        yhat[t] = fc[-1]
    return yhat


def _train_end(t, horizon):
    return max(0, t - horizon + 1)


def run_gbm(kind, X, y, n_min, horizon, refit_every=8):
    yhat = np.full(len(y), np.nan)
    model = None
    last = -10**9
    for t in range(n_min, len(y)):
        te = _train_end(t, horizon)
        if te < 20:
            continue
        if t - last >= refit_every or model is None:
            model = fit_lgbm(X[:te], y[:te]) if kind == "lgbm" else fit_xgb(X[:te], y[:te])
            last = t
        yhat[t] = float(np.asarray(model.predict(X[t : t + 1])).ravel()[0])
    return yhat, model


def run_lstm_wf(X, y, n_min, horizon, refit_every=16, seq_len=8):
    yhat = np.full(len(y), np.nan)
    model = None
    last = -10**9
    for t in range(max(n_min, seq_len), len(y)):
        te = _train_end(t, horizon)
        if te < seq_len + 8:
            continue
        if t - last >= refit_every or model is None:
            model = fit_lstm(X[:te], y[:te], seq_len=seq_len, epochs=8, hidden=8)
            last = t
        yhat[t] = predict_lstm(model, X[:te], seq_len=seq_len)
    return yhat


def forecast_next_model(
    name,
    price,
    X,
    y,
    x_next,
    Xex_full,
    vs_model=None,
    vs_scaler=None,
    seq_len=8,
):
    """Fit/update the selected h=1 model and forecast the next observed week."""
    price = np.asarray(price, dtype=float)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    x_next = np.asarray(x_next, dtype=float).reshape(1, -1)
    if name == "ARIMA":
        return float(arima_forecast(price, steps=1, model=None)[0][0]), {}
    if name == "ARIMAX":
        Xex_full = np.asarray(Xex_full, dtype=float)
        point = arimax_forecast(
            price,
            Xex_full,
            Xex_full[-1:],
            steps=1,
            model=None,
        )[0][0]
        return float(point), {}
    if name == "naive":
        return float(naive_predict(price, n=1)[0]), {}
    if name == "media_movel":
        return float(ma_predict(price, window=4, n=1)[0]), {}
    if name == "LightGBM":
        model = fit_lgbm(X, y)
        return float(np.asarray(model.predict(x_next)).ravel()[0]), {}
    if name == "XGBoost":
        model = fit_xgb(X, y)
        return float(np.asarray(model.predict(x_next)).ravel()[0]), {}
    if name == "LSTM":
        model = fit_lstm(X, y, seq_len=seq_len, epochs=8, hidden=8)
        point = predict_lstm(model, np.vstack([X, x_next]), seq_len=seq_len)
        return float(point), {}
    if name == "VS-ePL-KRLS":
        if vs_model is None or vs_scaler is None:
            raise ValueError("modelo e scaler VS-ePL-KRLS sao obrigatorios")
        # For h=1 the final shifted target has just become observable.
        vs_model.update(
            np.clip((X[-1] - vs_scaler[0]) / vs_scaler[1], -0.25, 1.25),
            float(y[-1]),
        )
        xt = np.clip((x_next.ravel() - vs_scaler[0]) / vs_scaler[1], -0.25, 1.25)
        return float(vs_model.predict_one(xt)), {"n_regras": int(vs_model.n_rules)}
    raise ValueError(f"modelo de producao nao suportado: {name}")


def eval_model(name, y, yhat, y_prev, n_min):
    mask = np.isfinite(yhat) & np.isfinite(y)
    mask[:n_min] = False
    m = summarize(y[mask], yhat[mask], y_prev[mask])
    m["model"] = name
    resid = (y - yhat)
    lo, hi = conformal_p10_p90(resid, yhat)
    interval_mask = mask & np.isfinite(lo) & np.isfinite(hi)
    m["coverage_p10_p90"] = (
        coverage(y[interval_mask], lo[interval_mask], hi[interval_mask])
        if interval_mask.any()
        else float("nan")
    )
    if interval_mask.sum() > 5:
        m["pinball10"] = pinball(y[interval_mask], lo[interval_mask], 0.1)
        m["pinball90"] = pinball(y[interval_mask], hi[interval_mask], 0.9)
    e_model = (y[mask] - yhat[mask])
    return m, resid, lo, hi, mask, e_model


def plot_forecast(dates, y, yhat, lo, hi, title, path):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(dates, y, color="black", lw=1.2, label="Real")
    ax.plot(dates, yhat, color="C0", lw=1.2, label="Previsto")
    if lo is not None:
        ax.fill_between(dates, lo, hi, color="C0", alpha=0.2, label="P10-P90")
    ax.set_title(title)
    ax.set_ylabel("R$/L")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    REP.mkdir(parents=True, exist_ok=True)
    all_rows = []
    prod_payload = {}
    for horizon in (1, 2, 4):
        print(f"\n=== Horizonte {horizon} semana(s) ===")
        df = load_panel(horizon)
        print(f"  amostras={len(df)} features={len(df.attrs.get('feat_cols', []))} {df['data'].min().date()}->{df['data'].max().date()}")
        feat_cols = df.attrs.get("feat_cols", FEATURE_COLS)
        leaks = leak_check(df, feat_cols)
        if leaks:
            raise RuntimeError(f"Vazamento de features: {leaks}")
        df[feat_cols] = df[feat_cols].ffill().bfill()
        X = df[feat_cols].to_numpy(float)
        Xex = df[ARIMAX_COLS].to_numpy(float)
        y = df["y"].to_numpy(float)
        y_prev = df["y_prev"].to_numpy(float)
        dates = df["data"]
        n_min = max(80, 12 + 8)
        preds = {}
        print("  VS-ePL-KRLS...")
        preds["VS-ePL-KRLS"], vs_model, vs_scaler = run_vsepl(X, y, n_min, horizon)
        print("  naive / media movel...")
        preds["naive"] = run_naive(y_prev, n_min)
        preds["media_movel"] = run_ma(y_prev, n_min, 4)
        print("  ARIMA...")
        preds["ARIMA"] = run_arima(y_prev, n_min, horizon, 26)
        print("  ARIMAX...")
        preds["ARIMAX"] = run_arimax(y_prev, Xex, n_min, horizon, 26)
        print("  LightGBM...")
        preds["LightGBM"], lgb_model = run_gbm("lgbm", X, y, n_min, horizon, 12)
        print("  XGBoost...")
        preds["XGBoost"], xgb_model = run_gbm("xgb", X, y, n_min, horizon, 12)
        print("  LSTM (subprocess)...")
        tmp = RES / f"_tmp_lstm_h{horizon}.npz"
        np.savez(tmp, X=X, y=y)
        yhat_path = tmp.with_name(tmp.stem + "_yhat.npy")
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "_lstm_wf.py"), str(tmp), str(n_min), str(horizon), "26", "8"],
            cwd=str(ROOT),
        )
        if r.returncode == 0 and yhat_path.exists():
            preds["LSTM"] = np.load(yhat_path)
        else:
            print(f"  LSTM indisponivel (exit={r.returncode}); segue sem esse benchmark.")

        naive_err = None
        eval_artifacts = {}
        for name, yhat in preds.items():
            m, resid, lo, hi, mask, e = eval_model(name, y, yhat, y_prev, n_min)
            m["horizon"] = horizon
            if name == "naive":
                naive_err = e
            if naive_err is not None and name != "naive" and len(e) == len(naive_err):
                dm = diebold_mariano(e, naive_err, h=horizon)
                m["dm_vs_naive"] = dm["dm_stat"]
                m["dm_p"] = dm["pvalue"]
            all_rows.append(m)
            eval_artifacts[name] = {"resid": resid, "mask": mask}
            print(f"    {name:12s} RMSE={m['rmse']:.4f} MAE={m['mae']:.4f} sMAPE={m['smape']:.2f} dir={m.get('dir_acc', np.nan):.3f}")
            if name == "VS-ePL-KRLS":
                plot_forecast(
                    dates[mask], y[mask], yhat[mask], lo[mask], hi[mask],
                    f"VS-ePL-KRLS semanal h={horizon}",
                    FIG / f"semanal_h{horizon}_vsepl.png",
                )
                ph = page_hinkley(resid[mask])
                rrmse = rolling_rmse(y, yhat, 12)
                feat_psi = {c: psi(X[:n_min, j], X[n_min:, j]) for j, c in enumerate(feat_cols)}
                (RES / f"drift_h{horizon}.json").write_text(
                    json.dumps({
                        "page_hinkley_alerts": int(ph["flags"].sum()),
                        "psi": feat_psi,
                        "n_rules_final": vs_model.n_rules,
                        "beta_final": vs_model.beta,
                    }, indent=2),
                    encoding="utf-8",
                )
                fig, ax = plt.subplots(figsize=(10, 3.5))
                ax.plot(dates, rrmse)
                ax.set_title(f"RMSE movel 12 semanas — h={horizon}")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(FIG / f"drift_rmse_h{horizon}.png", dpi=140)
                plt.close(fig)

        # permutation importance on LightGBM last model
        if lgb_model is not None:
            def pred(Xm):
                return np.asarray(lgb_model.predict(Xm), dtype=float)
            imp = permutation_importance(pred, X[n_min:], y[n_min:], feat_cols, n_repeats=3)
            (RES / f"importancia_lgbm_h{horizon}.json").write_text(json.dumps(imp, indent=2), encoding="utf-8")
            fig, ax = plt.subplots(figsize=(8, 5))
            names = list(imp.keys())[:12][::-1]
            vals = [imp[k] for k in names]
            ax.barh(names, vals)
            ax.set_title(f"Importancia por permutacao (LightGBM) h={horizon}")
            fig.tight_layout()
            fig.savefig(FIG / f"importancia_h{horizon}.png", dpi=140)
            plt.close(fig)

        if horizon == 1:
            full = pd.read_csv(PROC / "semanal_s10_features.csv", parse_dates=["data"])
            full[feat_cols] = full[feat_cols].ffill().bfill()
            full = full.dropna(subset=feat_cols).sort_values("data")
            x_last = full[feat_cols].to_numpy(float)[-1]
            h1_rows = [row for row in all_rows if row["horizon"] == 1]
            selected = min(h1_rows, key=lambda row: row["rmse"])["model"]
            yhat_next, model_meta = forecast_next_model(
                selected,
                full["revenda"].to_numpy(float),
                X,
                y,
                x_last,
                full[ARIMAX_COLS].to_numpy(float),
                vs_model=vs_model,
                vs_scaler=vs_scaler,
            )
            selected_resid = eval_artifacts[selected]["resid"]
            hist = selected_resid[np.isfinite(selected_resid)]
            q10, q90 = np.quantile(hist[-80:], [0.10, 0.90]) if len(hist) >= 20 else (np.nan, np.nan)
            last_price = float(full["revenda"].iloc[-1])
            probs = direction_probs(hist[-80:] if len(hist) else hist, yhat_next, last_price)
            prod_payload = {
                "modelo": selected,
                "ultima_semana_observada": str(pd.Timestamp(full["data"].iloc[-1]).date()),
                "preco_observado_ultima_semana": last_price,
                "horizonte": "1 semana",
                "previsao_pontual": float(yhat_next),
                "p10": float(yhat_next + q10) if np.isfinite(q10) else None,
                "p90": float(yhat_next + q90) if np.isfinite(q90) else None,
                "probabilidades": probs,
                **model_meta,
                "aviso": "Previsao do preco medio nacional de REVENDA. Nao e preco de bomba de um posto especifico.",
            }
            (RES / "previsao_proxima_semana.json").write_text(
                json.dumps(prod_payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    table = pd.DataFrame(all_rows)
    table.to_csv(RES / "semanal_benchmarks.csv", index=False)
    best = (
        table.sort_values(["horizon", "rmse"])
        .groupby("horizon", as_index=False)
        .first()[["horizon", "model", "rmse", "mae", "smape", "dir_acc"]]
    )
    lines = [
        "# Bloco 2 — Adaptacao semanal",
        "",
        "Validacao walk-forward temporal (sem divisao aleatoria).",
        "VS-ePL-KRLS atualiza de forma incremental. Modelos em lote reajustam a cada 4 semanas (LSTM a cada 8).",
        "Features apenas defasadas. Preco de distribuicao NAO entra no modelo de producao apos ago/2020.",
        "",
        "## Resultados",
        "",
        md_table(table[["horizon", "model", "rmse", "mae", "smape", "dir_acc", "coverage_p10_p90"]]),
        "",
        "## Melhor por horizonte (RMSE)",
        "",
        md_table(best),
        "",
        "## Previsao 1 semana a frente (ultimo ponto da amostra)",
        "",
        "```json",
        json.dumps(prod_payload, indent=2, ensure_ascii=False),
        "```",
        "",
        "Os numeros do bloco 1 (artigo mensal 2012-2020) nao se transferem para este bloco.",
    ]
    (REP / "02_semanal.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nRelatorio:", REP / "02_semanal.md")
    if prod_payload:
        print("\nPrevisao proxima semana:")
        print(json.dumps(prod_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
