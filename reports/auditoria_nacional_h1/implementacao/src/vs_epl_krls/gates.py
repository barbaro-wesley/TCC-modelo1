"""Gates decidiveis para a serie semanal do Diesel B S10.

Por que este modulo existe
--------------------------
O gate historico do projeto era "ganhar 2% de RMSE e ter Diebold-Mariano com
``p < 0,05``".  Ele nunca conclui nesta serie, e a causa esta medida: no holdout
de 104 semanas uma unica semana responde por 75,1% do erro quadratico do ARIMA e
tres semanas respondem por 91,2%.  O RMSE tem tamanho amostral efetivo de cerca
de tres, e o DM assintotico normal esta sendo interrogado sobre uma media cuja
distribuicao amostral nao chegou perto da normalidade.  O resultado pratico foi
um projeto que acumulou evidencia por meses sem conseguir promover nem descartar
nada.

O que muda aqui
---------------
Quatro trocas, todas na direcao de reduzir a influencia da cauda:

1. **MAE no lugar do RMSE** como metrica de acuracia primaria.  O erro absoluto
   tem cauda muito mais leve que o quadratico; a mesma semana extrema que
   domina 75% do MSE responde por uma fracao bem menor do MAE.
2. **Bootstrap em blocos moveis no lugar da aproximacao assintotica.**  Blocos
   preservam a dependencia semanal e nao exigem normalidade da media.  O teste
   DM continua disponivel, mas com a correcao de amostra pequena de
   Harvey, Leybourne e Newbold (1997) e referencia ``t`` com ``T-1`` graus de
   liberdade, nao normal.
3. **Metricas decompostas por regime.**  Em dois tercos das semanas nada
   acontece e a persistencia ganha de todos os modelos; agregar regimes esconde
   o unico comportamento que importa.  Semana parada e semana de evento sao
   avaliadas separadamente.
4. **Economia liquida da politica de compra como KPI primario**, com limite
   inferior de bootstrap, e o *interval score* de Winkler para a incerteza.  O
   produto decide comprar ou nao comprar; e nessa moeda que a decisao deve ser
   defendida.

Nada aqui escolhe modelo.  O modulo pontua previsoes que ja existem.

Referencias
-----------
- Harvey, Leybourne e Newbold (1997), correcao de amostra pequena do DM:
  ``S* = S * sqrt((T + 1 - 2h + h(h-1)/T) / T)``, comparado a ``t(T-1)``.
- Gneiting e Raftery (2007) e Winkler (1972), *interval score*:
  ``(u - l) + (2/alpha)*(l - y)*1{y<l} + (2/alpha)*(y - u)*1{y>u}``.
- Politis e Romano (1994), bootstrap em blocos para series dependentes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "EVENT_THRESHOLD_BRL",
    "AccuracyReport",
    "IntervalReport",
    "GateOutcome",
    "ChallengerVerdict",
    "winkler_score",
    "interval_report",
    "moving_block_bootstrap",
    "bootstrap_mean_ci",
    "paired_block_bootstrap",
    "diebold_mariano_hln",
    "event_mask",
    "accuracy_report",
    "evaluate_challenger",
]

#: Variacao semanal, em R$/L, a partir da qual a semana e tratada como evento.
#: E o mesmo limiar usado pelos experimentos de repasse e paridade.
EVENT_THRESHOLD_BRL = 0.02

#: Tamanho de bloco padrao do bootstrap, em semanas.  Quatro semanas cobrem o
#: repasse tipico de custo observado nesta serie sem encurtar demais o numero de
#: blocos independentes disponiveis num holdout de 104 pontos.
DEFAULT_BLOCK_WEEKS = 4


def _finite_pairs(*arrays: ArrayLike) -> tuple[NDArray[np.float64], ...]:
    """Alinha arrays e mantem apenas as posicoes finitas em todos eles."""

    values = [np.asarray(array, dtype=float).ravel() for array in arrays]
    if not values or values[0].size == 0:
        raise ValueError("as series de avaliacao nao podem ser vazias")
    sizes = {value.size for value in values}
    if len(sizes) != 1:
        raise ValueError(f"as series precisam ter o mesmo tamanho, recebi {sorted(sizes)}")
    keep = np.ones(values[0].size, dtype=bool)
    for value in values:
        keep &= np.isfinite(value)
    if not keep.any():
        raise ValueError("nenhuma observacao finita em comum entre as series")
    return tuple(value[keep] for value in values)


def event_mask(
    actual: ArrayLike,
    origin: ArrayLike,
    *,
    threshold: float = EVENT_THRESHOLD_BRL,
) -> NDArray[np.bool_]:
    """Marca as semanas em que o preco realmente se moveu.

    ``origin`` e o preco vigente na origem da previsao, ou seja, a previsao da
    persistencia.  Semanas cuja variacao absoluta nao passa de ``threshold`` sao
    "paradas": modelar nelas so acrescenta ruido, e e por isso que elas precisam
    ser avaliadas em separado.
    """

    truth, base = _finite_pairs(actual, origin)
    if threshold < 0:
        raise ValueError("threshold precisa ser nao negativo")
    return np.abs(truth - base) > float(threshold)


def winkler_score(
    actual: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    alpha: float,
) -> NDArray[np.float64]:
    """*Interval score* de Winkler por observacao; menor e melhor.

    Regra de pontuacao propria: premia intervalo estreito, mas cobra
    ``2/alpha`` por unidade de distancia sempre que o realizado cai fora.  Um
    intervalo largo demais e um intervalo que erra sao penalizados pela mesma
    escala, o que e exatamente o que falta a uma taxa de cobertura isolada.
    """

    truth, low, high = _finite_pairs(actual, lower, upper)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha precisa estar em (0, 1)")
    if np.any(high < low):
        raise ValueError("upper nao pode ser menor que lower")
    width = high - low
    below = np.where(truth < low, (2.0 / alpha) * (low - truth), 0.0)
    above = np.where(truth > high, (2.0 / alpha) * (truth - high), 0.0)
    return width + below + above


@dataclass(frozen=True)
class IntervalReport:
    """Qualidade do intervalo: cobertura, largura e pontuacao propria."""

    nominal_coverage: float
    empirical_coverage: float
    calibration_error: float
    mean_width: float
    median_width: float
    mean_winkler: float
    n: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def interval_report(
    actual: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    nominal_coverage: float,
) -> IntervalReport:
    """Pontua um intervalo previsto contra o realizado."""

    truth, low, high = _finite_pairs(actual, lower, upper)
    if not 0.0 < nominal_coverage < 1.0:
        raise ValueError("nominal_coverage precisa estar em (0, 1)")
    alpha = 1.0 - nominal_coverage
    covered = (truth >= low) & (truth <= high)
    empirical = float(np.mean(covered))
    scores = winkler_score(truth, low, high, alpha=alpha)
    width = high - low
    return IntervalReport(
        nominal_coverage=float(nominal_coverage),
        empirical_coverage=empirical,
        calibration_error=float(empirical - nominal_coverage),
        mean_width=float(np.mean(width)),
        median_width=float(np.median(width)),
        mean_winkler=float(np.mean(scores)),
        n=int(truth.size),
    )


def moving_block_bootstrap(
    values: ArrayLike,
    *,
    block_size: int = DEFAULT_BLOCK_WEEKS,
    n_resamples: int = 2000,
    random_state: int = 42,
) -> NDArray[np.float64]:
    """Reamostra ``values`` em blocos moveis e devolve as medias reamostradas.

    Blocos preservam a dependencia de curto prazo que o bootstrap i.i.d.
    destruiria.  E a alternativa honesta ao erro-padrao assintotico numa serie
    cuja media e dominada por poucos pontos.
    """

    (series,) = _finite_pairs(values)
    if n_resamples < 1:
        raise ValueError("n_resamples precisa ser positivo")
    block = int(min(max(1, int(block_size)), series.size))
    starts = np.arange(series.size - block + 1)
    blocks_needed = int(np.ceil(series.size / block))
    rng = np.random.default_rng(random_state)
    chosen = rng.choice(starts, size=(n_resamples, blocks_needed), replace=True)
    offsets = np.arange(block)
    # (n_resamples, blocks_needed, block) -> concatena e corta no tamanho original
    index = (chosen[:, :, None] + offsets[None, None, :]).reshape(n_resamples, -1)
    return series[index[:, : series.size]].mean(axis=1)


def bootstrap_mean_ci(
    values: ArrayLike,
    *,
    level: float = 0.90,
    block_size: int = DEFAULT_BLOCK_WEEKS,
    n_resamples: int = 2000,
    random_state: int = 42,
) -> tuple[float, float]:
    """Intervalo percentil de bootstrap em blocos para a media."""

    if not 0.0 < level < 1.0:
        raise ValueError("level precisa estar em (0, 1)")
    means = moving_block_bootstrap(
        values,
        block_size=block_size,
        n_resamples=n_resamples,
        random_state=random_state,
    )
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(low), float(high)


def paired_block_bootstrap(
    loss_challenger: ArrayLike,
    loss_incumbent: ArrayLike,
    *,
    block_size: int = DEFAULT_BLOCK_WEEKS,
    n_resamples: int = 2000,
    random_state: int = 42,
) -> dict[str, float]:
    """Compara duas perdas pareadas sem supor normalidade da media.

    Devolve a diferenca media (negativa favorece o challenger), o intervalo de
    90% e a probabilidade de bootstrap de o challenger nao ser melhor.  A
    reamostragem e pareada por bloco: as duas series andam juntas, entao a
    comparacao nao inventa ganho a partir de dessincronizacao.
    """

    challenger, incumbent = _finite_pairs(loss_challenger, loss_incumbent)
    difference = challenger - incumbent
    means = moving_block_bootstrap(
        difference,
        block_size=block_size,
        n_resamples=n_resamples,
        random_state=random_state,
    )
    low, high = np.quantile(means, [0.05, 0.95])
    return {
        "mean_difference": float(np.mean(difference)),
        "ci90_low": float(low),
        "ci90_high": float(high),
        # Fracao das reamostragens em que o challenger nao melhora.  E a leitura
        # direta de "qual a chance de este ganho ser sorte da janela".
        "pvalue_not_better": float(np.mean(means >= 0.0)),
        "n": int(difference.size),
    }


def diebold_mariano_hln(
    errors_challenger: ArrayLike,
    errors_incumbent: ArrayLike,
    *,
    horizon: int = 1,
    loss: str = "absolute",
) -> dict[str, float]:
    """Teste DM com a correcao de amostra pequena de Harvey-Leybourne-Newbold.

    O DM original compara a estatistica com a normal padrao, o que e
    generosamente otimista em 104 pontos dependentes e de cauda pesada.  A
    correcao multiplica a estatistica por
    ``sqrt((T + 1 - 2h + h(h-1)/T) / T)`` e usa ``t`` com ``T-1`` graus de
    liberdade.  ``loss="absolute"`` e o padrao aqui de proposito: no erro
    quadratico esta serie nao tem tamanho amostral efetivo para o teste.
    """

    from scipy import stats  # importado aqui: o modulo base nao exige scipy

    challenger, incumbent = _finite_pairs(errors_challenger, errors_incumbent)
    if horizon < 1:
        raise ValueError("horizon precisa ser positivo")
    if loss == "absolute":
        difference = np.abs(challenger) - np.abs(incumbent)
    elif loss == "squared":
        difference = challenger**2 - incumbent**2
    else:
        raise ValueError("loss precisa ser 'absolute' ou 'squared'")

    n = difference.size
    if n < 3:
        raise ValueError("o teste DM precisa de pelo menos tres observacoes")
    mean_difference = float(np.mean(difference))
    variance = float(np.var(difference, ddof=1))
    for lag in range(1, horizon):
        covariance = float(np.cov(difference[lag:], difference[:-lag], ddof=1)[0, 1])
        variance += 2.0 * (1.0 - lag / horizon) * covariance
    if variance <= 0.0:
        return {
            "dm_stat": 0.0,
            "dm_stat_hln": 0.0,
            "pvalue": 1.0,
            "mean_difference": mean_difference,
            "n": int(n),
        }
    statistic = mean_difference / np.sqrt(variance / n)
    correction = np.sqrt(
        (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
    )
    corrected = statistic * correction
    pvalue = 2.0 * (1.0 - stats.t.cdf(abs(corrected), df=n - 1))
    return {
        "dm_stat": float(statistic),
        "dm_stat_hln": float(corrected),
        "pvalue": float(pvalue),
        "mean_difference": mean_difference,
        "n": int(n),
    }


@dataclass(frozen=True)
class AccuracyReport:
    """Acuracia decomposta por regime, com MAE em primeiro lugar."""

    mae: float
    rmse: float
    mae_quiet: float
    mae_event: float
    rmse_quiet: float
    rmse_event: float
    directional_accuracy: float
    n: int
    n_quiet: int
    n_event: int
    largest_error_share_of_sse: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _safe_mean(values: NDArray[np.float64]) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def _safe_rmse(values: NDArray[np.float64]) -> float:
    return float(np.sqrt(np.mean(values**2))) if values.size else float("nan")


def accuracy_report(
    actual: ArrayLike,
    prediction: ArrayLike,
    origin: ArrayLike,
    *,
    threshold: float = EVENT_THRESHOLD_BRL,
) -> AccuracyReport:
    """Pontua uma previsao, separando semana parada de semana de evento.

    ``largest_error_share_of_sse`` mede a concentracao que motivou este modulo:
    e a fracao do erro quadratico total que vem do pior ponto.  Quando ela passa
    de 0,5, qualquer conclusao baseada em RMSE esta sendo decidida por uma
    observacao.
    """

    truth, forecast, base = _finite_pairs(actual, prediction, origin)
    errors = truth - forecast
    is_event = np.abs(truth - base) > float(threshold)
    squared = errors**2
    total = float(np.sum(squared))
    moved = truth != base
    directional = (
        float(np.mean(np.sign(forecast[moved] - base[moved]) == np.sign(truth[moved] - base[moved])))
        if moved.any()
        else float("nan")
    )
    return AccuracyReport(
        mae=_safe_mean(np.abs(errors)),
        rmse=_safe_rmse(errors),
        mae_quiet=_safe_mean(np.abs(errors[~is_event])),
        mae_event=_safe_mean(np.abs(errors[is_event])),
        rmse_quiet=_safe_rmse(errors[~is_event]),
        rmse_event=_safe_rmse(errors[is_event]),
        directional_accuracy=directional,
        n=int(truth.size),
        n_quiet=int((~is_event).sum()),
        n_event=int(is_event.sum()),
        largest_error_share_of_sse=float(np.max(squared) / total) if total > 0 else 0.0,
    )


@dataclass(frozen=True)
class GateOutcome:
    """Um criterio, o que ele mediu e se concluiu."""

    name: str
    passed: bool
    observed: float
    threshold: float
    rationale: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ChallengerVerdict:
    """Veredito completo, com os numeros que o sustentam."""

    challenger: str
    incumbent: str
    promote: bool
    gates: tuple[GateOutcome, ...] = field(default_factory=tuple)
    accuracy_challenger: AccuracyReport | None = None
    accuracy_incumbent: AccuracyReport | None = None
    interval_challenger: IntervalReport | None = None
    interval_incumbent: IntervalReport | None = None
    mae_comparison: dict[str, float] = field(default_factory=dict)
    dm_absolute: dict[str, float] = field(default_factory=dict)

    @property
    def failed_gates(self) -> tuple[str, ...]:
        return tuple(gate.name for gate in self.gates if not gate.passed)

    def as_dict(self) -> dict[str, object]:
        return {
            "challenger": self.challenger,
            "incumbent": self.incumbent,
            "promote": self.promote,
            "failed_gates": list(self.failed_gates),
            "gates": [gate.as_dict() for gate in self.gates],
            "accuracy_challenger": (
                self.accuracy_challenger.as_dict() if self.accuracy_challenger else None
            ),
            "accuracy_incumbent": (
                self.accuracy_incumbent.as_dict() if self.accuracy_incumbent else None
            ),
            "interval_challenger": (
                self.interval_challenger.as_dict() if self.interval_challenger else None
            ),
            "interval_incumbent": (
                self.interval_incumbent.as_dict() if self.interval_incumbent else None
            ),
            "mae_comparison": self.mae_comparison,
            "dm_absolute": self.dm_absolute,
        }


def evaluate_challenger(
    *,
    challenger: str,
    incumbent: str,
    actual: ArrayLike,
    origin: ArrayLike,
    prediction_challenger: ArrayLike,
    prediction_incumbent: ArrayLike,
    savings_challenger: ArrayLike | None = None,
    savings_incumbent: ArrayLike | None = None,
    interval_challenger: tuple[ArrayLike, ArrayLike] | None = None,
    interval_incumbent: tuple[ArrayLike, ArrayLike] | None = None,
    nominal_coverage: float = 0.80,
    max_event_mae_regression: float = 0.05,
    max_largest_event_share: float = 0.60,
    coverage_tolerance: float = 0.10,
    random_state: int = 42,
) -> ChallengerVerdict:
    """Aplica os gates decidiveis e devolve um veredito com os numeros.

    A ordem dos criterios e deliberada.  Economia liquida vem primeiro porque e
    a moeda em que o produto e defendido; acuracia vem depois, medida em MAE e
    julgada por bootstrap em blocos; regime e concentracao entram como travas
    contra ganho que vem de um unico ponto; e o intervalo e cobrado pelo Winkler,
    nao apenas pela taxa de cobertura.

    Todos os limiares sao argumentos: a funcao nao esconde politica dentro de
    numero magico.
    """

    accuracy_c = accuracy_report(actual, prediction_challenger, origin)
    accuracy_i = accuracy_report(actual, prediction_incumbent, origin)
    truth, base, forecast_c, forecast_i = _finite_pairs(
        actual, origin, prediction_challenger, prediction_incumbent
    )
    absolute_c = np.abs(truth - forecast_c)
    absolute_i = np.abs(truth - forecast_i)
    mae_comparison = paired_block_bootstrap(
        absolute_c, absolute_i, random_state=random_state
    )
    dm_absolute = diebold_mariano_hln(truth - forecast_c, truth - forecast_i)

    gates: list[GateOutcome] = []

    if savings_challenger is not None:
        weekly_c = np.asarray(savings_challenger, dtype=float).ravel()
        low, _ = bootstrap_mean_ci(weekly_c * 52.0, random_state=random_state)
        gates.append(
            GateOutcome(
                name="economia_anual_ci90_positiva",
                passed=bool(low > 0.0),
                observed=float(low),
                threshold=0.0,
                rationale=(
                    "limite inferior de 90% da economia anualizada, bootstrap em blocos; "
                    "o KPI primario precisa ser positivo mesmo na reamostragem pessimista"
                ),
            )
        )
        if savings_incumbent is not None:
            weekly_i = np.asarray(savings_incumbent, dtype=float).ravel()
            gates.append(
                GateOutcome(
                    name="economia_supera_incumbente",
                    passed=bool(np.sum(weekly_c) > np.sum(weekly_i)),
                    observed=float(np.sum(weekly_c) - np.sum(weekly_i)),
                    threshold=0.0,
                    rationale="economia liquida acumulada acima da do modelo em producao",
                )
            )
        total = float(np.sum(weekly_c))
        share = float(np.max(weekly_c) / total) if total > 0 and weekly_c.size else 1.0
        gates.append(
            GateOutcome(
                name="economia_nao_concentrada",
                passed=bool(share <= max_largest_event_share),
                observed=share,
                threshold=float(max_largest_event_share),
                rationale=(
                    "fracao da economia vinda do maior evento; acima do limite o numero "
                    "e uma aposta num episodio, nao uma politica"
                ),
            )
        )

    gates.append(
        GateOutcome(
            name="mae_melhor_que_incumbente",
            passed=bool(mae_comparison["ci90_high"] < 0.0),
            observed=float(mae_comparison["ci90_high"]),
            threshold=0.0,
            rationale=(
                "topo do IC90 da diferenca de MAE pareada por bloco; exige que ate a "
                "reamostragem pessimista favoreca o challenger"
            ),
        )
    )

    event_regression = (
        float(accuracy_c.mae_event / accuracy_i.mae_event - 1.0)
        if np.isfinite(accuracy_i.mae_event) and accuracy_i.mae_event > 0
        else float("inf")
    )
    gates.append(
        GateOutcome(
            name="sem_regressao_em_semana_de_evento",
            passed=bool(event_regression <= max_event_mae_regression),
            observed=event_regression,
            threshold=float(max_event_mae_regression),
            rationale=(
                "piora relativa do MAE nas semanas em que o preco realmente se moveu; "
                "e onde a decisao de compra acontece"
            ),
        )
    )

    quiet_regression = (
        float(accuracy_c.mae_quiet / accuracy_i.mae_quiet - 1.0)
        if np.isfinite(accuracy_i.mae_quiet) and accuracy_i.mae_quiet > 0
        else float("inf")
    )
    gates.append(
        GateOutcome(
            name="sem_regressao_em_semana_parada",
            passed=bool(quiet_regression <= max_event_mae_regression),
            observed=quiet_regression,
            threshold=float(max_event_mae_regression),
            rationale=(
                "piora relativa do MAE nas semanas paradas; modelo que agita preco "
                "parado gera gatilho falso e custo de carregamento"
            ),
        )
    )

    report_c: IntervalReport | None = None
    report_i: IntervalReport | None = None
    if interval_challenger is not None:
        report_c = interval_report(
            actual,
            interval_challenger[0],
            interval_challenger[1],
            nominal_coverage=nominal_coverage,
        )
        gates.append(
            GateOutcome(
                name="intervalo_calibrado",
                passed=bool(abs(report_c.calibration_error) <= coverage_tolerance),
                observed=float(report_c.calibration_error),
                threshold=float(coverage_tolerance),
                rationale=(
                    "erro de calibracao do intervalo; cobertura muito acima do nominal "
                    "e intervalo largo demais para decidir, nao seguranca"
                ),
            )
        )
        if interval_incumbent is not None:
            report_i = interval_report(
                actual,
                interval_incumbent[0],
                interval_incumbent[1],
                nominal_coverage=nominal_coverage,
            )
            gates.append(
                GateOutcome(
                    name="winkler_nao_pior",
                    passed=bool(report_c.mean_winkler <= report_i.mean_winkler),
                    observed=float(report_c.mean_winkler - report_i.mean_winkler),
                    threshold=0.0,
                    rationale=(
                        "interval score de Winkler contra o incumbente; cobra largura e "
                        "furo na mesma escala"
                    ),
                )
            )

    return ChallengerVerdict(
        challenger=challenger,
        incumbent=incumbent,
        promote=all(gate.passed for gate in gates),
        gates=tuple(gates),
        accuracy_challenger=accuracy_c,
        accuracy_incumbent=accuracy_i,
        interval_challenger=report_c,
        interval_incumbent=report_i,
        mae_comparison=mae_comparison,
        dm_absolute=dm_absolute,
    )
