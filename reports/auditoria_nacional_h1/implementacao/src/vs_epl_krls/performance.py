"""As tres taxas que um comprador pergunta, definidas sem ambiguidade.

Por que definir antes de medir
------------------------------
"Taxa de acerto" nao e uma metrica, sao tres perguntas diferentes, e cada uma
admite varias respostas conforme a convencao adotada.  Reportar um numero sem a
definicao ao lado e como reportar uma temperatura sem a escala.  Este modulo fixa
as convencoes:

**1. Lucratividade.**  A politica de antecipacao dispara algumas vezes por ano.
Sobre esses disparos medimos, na convencao de mercado: quantos deram lucro
(*taxa de acerto*), quanto o total ganho supera o total perdido (*fator de
lucro*), quanto rende um disparo medio (*expectativa*) e quanto tudo isso
representa sobre o gasto com combustivel (*retorno sobre o gasto*).  O ultimo e
o unico que responde "vale a pena?" — os outros tres explicam de onde ele vem.

**2. Acerto de movimento.**  Percentual de semanas em que o modelo acertou a
*direcao*.  Reportado em dois recortes, porque eles divergem: sobre todas as
semanas em que o preco se moveu, e sobre as semanas de evento — as que passam do
limiar e efetivamente decidem a compra.  Semana parada e excluida do calculo:
acertar que nada aconteceria nao e previsao, e incluir isso infla o numero.

**3. Acerto de preco.**  Nao existe resposta unica sem escolher uma tolerancia,
entao devolvemos a escada inteira: a fracao de semanas em que o erro ficou dentro
de 1, 2, 5 e 10 centavos por litro, mais a cobertura do intervalo P10-P90.  Quem
precisa de um numero so deve escolher a tolerancia que corresponde a decisao que
vai tomar, e dizer qual e.

Nada aqui e novo modelo: e leitura de previsoes que ja existem.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from .gates import EVENT_THRESHOLD_BRL, bootstrap_mean_ci

__all__ = [
    "PRICE_TOLERANCES_BRL",
    "MovementAccuracy",
    "PerformanceReport",
    "PriceAccuracy",
    "Profitability",
    "performance_report",
]

#: Tolerancias da escada de acerto de preco, em R$/L.  Um centavo e a ordem de
#: grandeza do limiar da politica de compra; dez centavos e a ordem de grandeza
#: de um reajuste de refinaria.
PRICE_TOLERANCES_BRL: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10)


def _aligned(*arrays: ArrayLike) -> tuple[np.ndarray, ...]:
    values = [np.asarray(array, dtype=float).ravel() for array in arrays]
    sizes = {value.size for value in values}
    if len(sizes) != 1 or values[0].size == 0:
        raise ValueError(f"as series precisam ter o mesmo tamanho nao nulo, recebi {sorted(sizes)}")
    keep = np.ones(values[0].size, dtype=bool)
    for value in values:
        keep &= np.isfinite(value)
    if not keep.any():
        raise ValueError("nenhuma observacao finita em comum")
    return tuple(value[keep] for value in values)


@dataclass(frozen=True)
class Profitability:
    """Resultado da politica de antecipacao, na convencao de mercado."""

    triggers: int
    winning_triggers: int
    #: Fracao dos disparos que deram lucro.
    win_rate: float | None
    #: Total ganho dividido pelo total perdido.  Acima de 1 significa que os
    #: acertos pagam os erros; abaixo, que nao pagam.
    profit_factor: float | None
    #: Resultado medio de um disparo, em R$.  E o numero que responde "vale a
    #: pena agir?" — win rate alto com expectativa negativa e armadilha comum.
    expectancy_brl: float | None
    gross_profit_brl: float
    gross_loss_brl: float
    net_savings_brl: float
    #: Economia liquida sobre o gasto total com combustivel no periodo.
    return_on_spend: float | None
    annualized_savings_brl: float | None
    annualized_ci90_brl: tuple[float, float] | None
    #: Fracao da economia vinda do maior evento isolado.
    largest_event_share: float | None
    weeks: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MovementAccuracy:
    """Acerto de direcao, separando o que decide compra do que nao decide."""

    #: Sobre todas as semanas em que o preco se moveu, por menos que fosse.
    hit_rate_moved: float | None
    n_moved: int
    #: Sobre as semanas de evento, que passam do limiar e decidem a compra.
    hit_rate_events: float | None
    n_events: int
    #: Sobre as semanas paradas, onde o modelo idealmente nao inventa movimento.
    hit_rate_quiet: float | None
    n_quiet: int
    threshold_brl: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PriceAccuracy:
    """Escada de acerto de preco, porque um numero so exigiria escolher por voce."""

    within: dict[str, float]
    median_absolute_error: float
    mean_absolute_error: float
    interval_coverage: float | None
    interval_nominal: float | None
    n: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PerformanceReport:
    model: str
    window: str
    profitability: Profitability | None = None
    movement: MovementAccuracy | None = None
    price: PriceAccuracy | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "window": self.window,
            "profitability": self.profitability.as_dict() if self.profitability else None,
            "movement": self.movement.as_dict() if self.movement else None,
            "price": self.price.as_dict() if self.price else None,
            "notes": list(self.notes),
        }


def _profitability(
    weekly_savings: np.ndarray,
    baseline_spend: float | None,
    triggers: int | None,
    winning_triggers: int | None,
) -> Profitability:
    gains = weekly_savings[weekly_savings > 0]
    losses = weekly_savings[weekly_savings < 0]
    gross_profit = float(gains.sum())
    gross_loss = float(-losses.sum())
    total = float(weekly_savings.sum())
    # A contagem de disparos vem da politica, nao da economia: um disparo cujo
    # preco nao se moveu rende exatamente zero e continua sendo um disparo.
    # Nesta serie dois tercos das semanas nao se movem, entao inferir disparo a
    # partir de economia nao nula subestima o denominador e infla a taxa de
    # acerto.
    n_triggers = int(triggers) if triggers is not None else int((weekly_savings != 0).sum())
    n_wins = int(winning_triggers) if winning_triggers is not None else int(gains.size)
    low, high = (
        bootstrap_mean_ci(weekly_savings * 52.0)
        if weekly_savings.size >= 8
        else (float("nan"), float("nan"))
    )
    largest = float(weekly_savings.max()) if weekly_savings.size else 0.0
    return Profitability(
        triggers=n_triggers,
        winning_triggers=n_wins,
        win_rate=float(n_wins / n_triggers) if n_triggers else None,
        profit_factor=(
            float(gross_profit / gross_loss)
            if gross_loss > 0
            else (float("inf") if gross_profit > 0 else None)
        ),
        expectancy_brl=float(total / n_triggers) if n_triggers else None,
        gross_profit_brl=gross_profit,
        gross_loss_brl=gross_loss,
        net_savings_brl=total,
        return_on_spend=(
            float(total / baseline_spend) if baseline_spend and baseline_spend > 0 else None
        ),
        annualized_savings_brl=(
            float(np.mean(weekly_savings) * 52.0) if weekly_savings.size else None
        ),
        annualized_ci90_brl=(low, high),
        largest_event_share=float(largest / total) if total > 0 else None,
        weeks=int(weekly_savings.size),
    )


def performance_report(
    *,
    model: str,
    window: str,
    actual: ArrayLike,
    prediction: ArrayLike,
    origin: ArrayLike,
    lower: ArrayLike | None = None,
    upper: ArrayLike | None = None,
    nominal_coverage: float | None = 0.80,
    weekly_savings: ArrayLike | None = None,
    triggers: int | None = None,
    winning_triggers: int | None = None,
    baseline_spend_brl: float | None = None,
    event_threshold: float = EVENT_THRESHOLD_BRL,
    notes: tuple[str, ...] = (),
) -> PerformanceReport:
    """Calcula as tres taxas para uma serie de previsoes ja emitidas."""

    truth, forecast, base = _aligned(actual, prediction, origin)
    error = truth - forecast
    realized_move = truth - base
    predicted_move = forecast - base

    moved = realized_move != 0.0
    events = np.abs(realized_move) > float(event_threshold)
    quiet = ~events
    correct = np.sign(predicted_move) == np.sign(realized_move)

    movement = MovementAccuracy(
        hit_rate_moved=float(np.mean(correct[moved])) if moved.any() else None,
        n_moved=int(moved.sum()),
        hit_rate_events=float(np.mean(correct[events])) if events.any() else None,
        n_events=int(events.sum()),
        hit_rate_quiet=(
            float(np.mean(correct[quiet & moved])) if (quiet & moved).any() else None
        ),
        n_quiet=int(quiet.sum()),
        threshold_brl=float(event_threshold),
    )

    coverage = None
    if lower is not None and upper is not None:
        low_band, high_band = _aligned(lower, upper)
        if low_band.size == truth.size:
            coverage = float(np.mean((truth >= low_band) & (truth <= high_band)))

    price = PriceAccuracy(
        within={
            f"{tolerance:.2f}": float(np.mean(np.abs(error) <= tolerance))
            for tolerance in PRICE_TOLERANCES_BRL
        },
        median_absolute_error=float(np.median(np.abs(error))),
        mean_absolute_error=float(np.mean(np.abs(error))),
        interval_coverage=coverage,
        interval_nominal=nominal_coverage if coverage is not None else None,
        n=int(truth.size),
    )

    profitability = None
    if weekly_savings is not None:
        savings = np.asarray(weekly_savings, dtype=float).ravel()
        savings = savings[np.isfinite(savings)]
        if savings.size:
            profitability = _profitability(
                savings, baseline_spend_brl, triggers, winning_triggers
            )

    return PerformanceReport(
        model=model,
        window=window,
        profitability=profitability,
        movement=movement,
        price=price,
        notes=notes,
    )
