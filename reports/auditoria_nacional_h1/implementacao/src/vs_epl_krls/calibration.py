"""Intervalos online calibrados por inferencia conformal adaptativa.

O problema medido
-----------------
O bundle de producao monta o intervalo P10-P90 com quantis de residuos em janela
movel.  Isso supoe que a distribuicao do erro de amanha se parece com a dos
ultimos tres anos.  Nesta serie ela nao se parece: o holdout de 104 semanas
entregou **cobertura de 92,3% para um intervalo nominal de 80%**.  Intervalo
conservador nao e seguranca gratuita.  Ele custa decisao: a politica de compra
compara a alta prevista com um limiar, e uma banda larga demais desloca o
cenario P90 para longe do plausivel, inflando o risco aparente de nao antecipar
e o custo aparente de antecipar.

A correcao
----------
Inferencia conformal adaptativa (Gibbs e Candes, 2021).  Em vez de fixar o nivel
de miscobertura, ele vira um parametro que aprende online::

    alpha(t+1) = alpha(t) + gamma * (alpha_alvo - 1{y(t) fora do intervalo})

Quando o intervalo cobre, ``alpha`` sobe e a banda encolhe; quando fura,
``alpha`` cai e a banda abre.  A garantia do metodo e de longo prazo e nao supoe
nada sobre a distribuicao dos dados: a cobertura empirica converge para
``1 - alpha_alvo`` mesmo sob mudanca arbitraria de regime.  E exatamente a
hipotese que uma serie de combustivel viola o tempo todo.

Residuo normalizado
-------------------
Os quantis sao calculados sobre o residuo *padronizado* ``e / s``, onde ``s`` e a
escala condicional que o modelo ja estima, e a banda volta para a escala do dia
multiplicando por ``s`` de novo.  Sem isso o calibrador jogaria fora a
heterocedasticidade que o modelo captura, e numa serie de saltos isso custa caro:
a banda ficaria estreita justamente nas semanas de choque.  Com ``scale=1`` o
comportamento e identico ao conformal simples.

Desvio deliberado em relacao ao artigo
--------------------------------------
No artigo, ``alpha(t) <= 0`` produz intervalo infinito.  Num produto isso e
inutil: um intervalo infinito nao informa decisao nenhuma.  Aqui ``alpha`` e
limitado a ``alpha_bounds`` e cada passo em que o limite foi atingido fica
registrado em ``clipped_steps``, para que a degradacao apareca no relatorio em
vez de sumir dentro de uma banda absurda.

Referencia
----------
Gibbs e Candes, "Adaptive Conformal Inference Under Distribution Shift",
NeurIPS 2021; e a extensao para deslocamento arbitrario em JMLR 25 (2024).
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "AdaptiveConformalInterval",
    "ConformalState",
    "backtest_adaptive_interval",
]


@dataclass(frozen=True)
class ConformalState:
    """Estado observavel do calibrador, para telemetria e auditoria."""

    nominal_coverage: float
    alpha_target: float
    alpha_current: float
    gamma: float
    residuals_in_window: int
    updates: int
    covered: int
    clipped_steps: int
    warmed_up: bool

    @property
    def running_coverage(self) -> float:
        return float(self.covered / self.updates) if self.updates else float("nan")

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = dict(asdict(self))
        payload["running_coverage"] = self.running_coverage
        return payload


class AdaptiveConformalInterval:
    """Intervalo preditivo que persegue a cobertura nominal online.

    Uso previsto, um passo por semana::

        band = AdaptiveConformalInterval(nominal_coverage=0.80)
        for point, sigma, truth in stream:
            lower, upper = band.interval(point, scale=sigma)   # decide agora
            band.observe(truth, lower, upper, residual=truth - point)

    ``interval`` nunca olha o futuro e ``observe`` so e chamado quando o valor
    oficial chega.  Enquanto a janela nao tem residuos suficientes, o objeto
    declara ``warmed_up=False`` e cai para o multiplo gaussiano de ``scale``, o
    mesmo comportamento defensivo do bundle atual.
    """

    def __init__(
        self,
        *,
        nominal_coverage: float = 0.80,
        gamma: float = 0.02,
        window: int = 156,
        min_residuals: int = 20,
        alpha_bounds: tuple[float, float] = (0.01, 0.60),
        normalize: bool = True,
    ) -> None:
        if not 0.0 < nominal_coverage < 1.0:
            raise ValueError("nominal_coverage precisa estar em (0, 1)")
        if gamma <= 0.0:
            raise ValueError("gamma precisa ser positivo")
        if window < 2:
            raise ValueError("window precisa ter ao menos duas posicoes")
        if min_residuals < 2:
            raise ValueError("min_residuals precisa ser ao menos dois")
        low, high = alpha_bounds
        if not 0.0 < low < high < 1.0:
            raise ValueError("alpha_bounds precisa satisfazer 0 < low < high < 1")

        self.nominal_coverage = float(nominal_coverage)
        self.alpha_target = 1.0 - float(nominal_coverage)
        self.gamma = float(gamma)
        self.window = int(window)
        self.min_residuals = int(min_residuals)
        self.alpha_bounds = (float(low), float(high))
        self.normalize = bool(normalize)
        self._alpha = float(np.clip(self.alpha_target, low, high))
        self._last_scale = 1.0
        self._residuals: deque[float] = deque(maxlen=self.window)
        self._updates = 0
        self._covered = 0
        self._clipped = 0

    # -- consulta ---------------------------------------------------------

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def warmed_up(self) -> bool:
        return len(self._residuals) >= self.min_residuals

    def state(self) -> ConformalState:
        return ConformalState(
            nominal_coverage=self.nominal_coverage,
            alpha_target=self.alpha_target,
            alpha_current=self._alpha,
            gamma=self.gamma,
            residuals_in_window=len(self._residuals),
            updates=self._updates,
            covered=self._covered,
            clipped_steps=self._clipped,
            warmed_up=self.warmed_up,
        )

    # -- operacao ---------------------------------------------------------

    def seed(self, residuals: ArrayLike) -> "AdaptiveConformalInterval":
        """Carrega residuos historicos sem contabiliza-los como cobertura."""

        values = np.asarray(residuals, dtype=float).ravel()
        for value in values[np.isfinite(values)]:
            self._residuals.append(float(value))
        return self

    def interval(self, point: float, *, scale: float = 1.0) -> tuple[float, float]:
        """Banda para ``point``; ``scale`` e o sigma condicional, se houver."""

        center = float(point)
        magnitude = abs(float(scale)) or 1.0
        if not np.isfinite(center):
            raise ValueError("point precisa ser finito")
        if not np.isfinite(magnitude) or magnitude <= 0.0:
            raise ValueError("scale precisa ser finito e positivo")
        self._last_scale = magnitude
        tail = self._alpha / 2.0
        if not self.warmed_up:
            # Fallback gaussiano ate a janela encher: mesma postura defensiva do
            # bundle em producao, e explicitamente sinalizado por warmed_up.
            from scipy import stats

            spread = float(stats.norm.ppf(1.0 - tail)) * magnitude
            return center - spread, center + spread
        residuals = np.fromiter(self._residuals, dtype=float)
        low, high = np.quantile(residuals, [tail, 1.0 - tail])
        if self.normalize:
            return center + float(low) * magnitude, center + float(high) * magnitude
        return center + float(low), center + float(high)

    def observe(
        self,
        actual: float,
        lower: float,
        upper: float,
        *,
        residual: float | None = None,
    ) -> ConformalState:
        """Atualiza ``alpha`` pela cobertura observada e guarda o residuo."""

        truth = float(actual)
        if not np.isfinite(truth):
            raise ValueError("actual precisa ser finito")
        covered = bool(float(lower) <= truth <= float(upper))
        self._updates += 1
        self._covered += int(covered)
        # O passo do artigo: erra para fora, alpha cai e a banda abre; cobre,
        # alpha sobe e a banda encolhe.
        proposed = self._alpha + self.gamma * (self.alpha_target - (0.0 if covered else 1.0))
        low, high = self.alpha_bounds
        clipped = float(np.clip(proposed, low, high))
        if clipped != proposed:
            self._clipped += 1
        self._alpha = clipped
        if residual is not None and np.isfinite(residual):
            # Guarda o residuo na mesma escala em que os quantis serao aplicados.
            divisor = self._last_scale if self.normalize else 1.0
            self._residuals.append(float(residual) / divisor)
        return self.state()


def backtest_adaptive_interval(
    actual: ArrayLike,
    point: ArrayLike,
    *,
    scale: ArrayLike | None = None,
    seed_residuals: ArrayLike | None = None,
    nominal_coverage: float = 0.80,
    gamma: float = 0.02,
    window: int = 156,
    min_residuals: int = 20,
    normalize: bool = True,
) -> dict[str, object]:
    """Roda o calibrador passo a passo e devolve as bandas emitidas.

    A ordem e estritamente causal: a banda de cada semana e emitida antes de o
    valor daquela semana ser observado.  O objeto calibrado volta em ``band``
    para que a producao continue de onde o backtest parou, em vez de recomecar
    com ``alpha`` no valor nominal.
    """

    truth = np.asarray(actual, dtype=float).ravel()
    center = np.asarray(point, dtype=float).ravel()
    if truth.shape != center.shape:
        raise ValueError("actual e point precisam ter o mesmo tamanho")
    if truth.size == 0:
        raise ValueError("as series nao podem ser vazias")
    magnitude = (
        np.ones_like(truth)
        if scale is None
        else np.asarray(scale, dtype=float).ravel()
    )
    if magnitude.shape != truth.shape:
        raise ValueError("scale precisa ter o mesmo tamanho de actual")

    band = AdaptiveConformalInterval(
        nominal_coverage=nominal_coverage,
        gamma=gamma,
        window=window,
        min_residuals=min_residuals,
        normalize=normalize,
    )
    if seed_residuals is not None:
        band.seed(seed_residuals)

    lower = np.empty_like(truth)
    upper = np.empty_like(truth)
    alphas = np.empty_like(truth)
    for index in range(truth.size):
        alphas[index] = band.alpha
        low, high = band.interval(center[index], scale=magnitude[index])
        lower[index] = low
        upper[index] = high
        if np.isfinite(truth[index]):
            band.observe(truth[index], low, high, residual=truth[index] - center[index])
    return {
        "lower": lower,
        "upper": upper,
        "alpha": alphas,
        "state": band.state(),
        "band": band,
    }
