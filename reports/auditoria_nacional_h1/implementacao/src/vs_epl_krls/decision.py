"""Camada de decisao: o que o comprador faz, nao apenas qual sera o preco.

Motivacao
---------
O produto ate aqui entregava um numero: a previsao da semana seguinte com um
intervalo.  Numero nao e decisao.  Quem compra 200 mil litros por mes precisa
saber se antecipa, quanto antecipa, quanto ganha se estiver certo, quanto perde
se estiver errado, e o quanto pode confiar nisso hoje.

Esta camada responde essas cinco perguntas a partir de artefatos que ja existem
e ja sao verificados por hash.  Ela nao treina, nao promove modelo e nao compra
nada: e leitura, agregacao e governanca explicita.

Governanca do sinal
-------------------
A recomendacao segue o **modelo primario da release**, que hoje e o ARIMA.  O
modelo de paridade entra como challenger visivel: quando os dois discordam da
direcao, a decisao nao muda de dono, mas a confianca cai para ``baixa`` e a
divergencia aparece no payload em vez de ser escondida atras de uma media.

Essa escolha nao e cautela generica, e o resultado medido pelos gates
decidiveis (``vs_epl_krls.gates``) sobre a evidencia publicada: o paridade passa
nos tres gates economicos, mas perde no MAE e piora 44% nas semanas paradas.  Um
modelo que agita preco parado gera gatilho falso, e gatilho falso custa
carregamento.  ``governance()`` devolve exatamente esses numeros, para que o
cliente possa discordar da decisao com dados na mao.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .monitoring import review_ledgers
from .product import S10ProductService, read_trusted_json

__all__ = [
    "DEFAULT_SIGNAL_THRESHOLD",
    "DEFAULT_FLEXIBILITY",
    "ModelView",
    "S10BasisReport",
    "S10Decision",
    "S10DecisionService",
]

#: Limiar pre-registrado da politica de antecipacao, em R$/L.  Nao foi ajustado
#: depois de escolhido: a varredura de sensibilidade mostrou que ja era o otimo.
DEFAULT_SIGNAL_THRESHOLD = 0.01

#: Fracao de uma semana de consumo que a politica antecipa quando dispara.
DEFAULT_FLEXIBILITY = 0.25


def _direction(change: float, threshold: float) -> str:
    if change > threshold:
        return "alta"
    if change < -threshold:
        return "queda"
    return "estavel"


@dataclass(frozen=True)
class ModelView:
    """O que um modelo diz sobre a semana que vem."""

    name: str
    role: str
    point_brl_per_liter: float
    expected_change_brl_per_liter: float
    direction: str
    would_trigger: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class S10BasisReport:
    """O que custa orcar pela media nacional em vez da serie do estado.

    E o numero que sustenta a conversa comercial, e ele **nao depende de modelo**:
    e a diferenca medida entre duas series publicas da ANP.
    """

    uf: str
    as_of: str
    volume_liters: float
    state_price_brl_per_liter: float
    national_price_brl_per_liter: float
    current_spread_brl_per_liter: float
    relative_difference: float
    mean_absolute_spread_brl_per_liter: float
    annual_budget_error_brl: float
    current_annual_gap_brl: float
    spread_z: float
    spread_percentile: float
    history_weeks: int
    window_weeks: int
    disclaimer: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class S10Decision:
    """Recomendacao acionavel, com o tamanho da aposta e o custo de errar."""

    target_date: str
    origin_date: str
    origin_price_brl_per_liter: float
    volume_liters: float
    recommendation: str
    liters_to_prebuy: float
    expected_saving_brl: float
    exposure_if_wrong_brl: float
    signal_threshold_brl_per_liter: float
    confidence: str
    models_agree: bool
    decided_by: str
    scope: str
    evidence_status: str
    views: tuple[ModelView, ...]
    interval_low_brl_per_liter: float
    interval_high_brl_per_liter: float
    interval_nominal_coverage: float
    rationale: tuple[str, ...]
    disclaimer: str

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["views"] = [view.as_dict() for view in self.views]
        payload["rationale"] = list(self.rationale)
        return payload


class S10DecisionService:
    """Combina release primaria e challenger numa recomendacao auditavel."""

    DISCLAIMER = (
        "Replay de politica sobre o preco medio nacional da ANP. Nao e cotacao de "
        "fornecedor, nao considera frete, contrato, estoque nem custo de capital, e "
        "nenhuma compra e executada automaticamente."
    )

    def __init__(
        self,
        product: S10ProductService,
        *,
        challenger_forecast: str | Path | None = None,
        gate_review: str | Path | None = None,
        regional_forecasts: Mapping[str, str | Path] | None = None,
        ledgers: Mapping[str, str | Path] | None = None,
        signal_threshold: float = DEFAULT_SIGNAL_THRESHOLD,
        flexibility: float = DEFAULT_FLEXIBILITY,
    ) -> None:
        if signal_threshold < 0:
            raise ValueError("signal_threshold nao pode ser negativo")
        if not 0.0 <= flexibility <= 1.0:
            raise ValueError("flexibility precisa estar entre zero e um")
        self.product = product
        self.signal_threshold = float(signal_threshold)
        self.flexibility = float(flexibility)
        self.challenger_path = (
            Path(challenger_forecast).resolve() if challenger_forecast else None
        )
        self.gate_review_path = Path(gate_review).resolve() if gate_review else None
        self.challenger = (
            read_trusted_json(self.challenger_path) if self.challenger_path else None
        )
        self.gate_manifest = (
            read_trusted_json(self.gate_review_path) if self.gate_review_path else None
        )
        self.regional = {
            str(uf).strip().upper(): read_trusted_json(Path(path).resolve())
            for uf, path in (regional_forecasts or {}).items()
        }
        # Caminhos, nao conteudo: o ledger e lido e verificado a cada consulta,
        # porque ele muda entre requisicoes enquanto a release nao muda.
        self.ledger_paths = {
            str(name): Path(path) for name, path in (ledgers or {}).items()
        }

    # ------------------------------------------------------------- estadual

    @property
    def available_states(self) -> tuple[str, ...]:
        return tuple(sorted(self.regional))

    def _regional_payload(self, uf: str) -> dict[str, Any]:
        code = str(uf).strip().upper()
        payload = self.regional.get(code)
        if payload is None:
            raise KeyError(
                f"nenhuma previsao estadual carregada para {code}; "
                f"disponiveis: {list(self.available_states)}"
            )
        target = str(payload["forecast"]["target_date"])
        expected = self.product.current_forecast.target_date
        if target != expected:
            # Servir um estado de outra semana-alvo faria o cliente comparar
            # numeros de periodos diferentes sem perceber.
            raise RuntimeError(
                f"a previsao de {code} aponta para {target}, mas a release "
                f"vigente e de {expected}"
            )
        return payload

    def basis(self, uf: str, volume_liters: float = 200_000.0) -> S10BasisReport:
        """Escala o relatorio de base pelo volume do cliente."""

        volume = float(volume_liters)
        if not 0 < volume <= 50_000_000:
            raise ValueError("volume_liters precisa ser positivo e no maximo 50 milhoes")
        payload = self._regional_payload(uf)
        basis = payload["basis"]
        annual_litres = volume * 12.0
        return S10BasisReport(
            uf=str(basis["uf"]),
            as_of=str(basis["as_of"]),
            volume_liters=volume,
            state_price_brl_per_liter=float(basis["state_price_brl_per_liter"]),
            national_price_brl_per_liter=float(basis["national_price_brl_per_liter"]),
            current_spread_brl_per_liter=float(basis["current_spread_brl_per_liter"]),
            relative_difference=float(basis["relative_difference"]),
            mean_absolute_spread_brl_per_liter=float(
                basis["mean_absolute_spread_brl_per_liter"]
            ),
            annual_budget_error_brl=round(
                float(basis["mean_absolute_spread_brl_per_liter"]) * annual_litres, 2
            ),
            current_annual_gap_brl=round(
                abs(float(basis["current_spread_brl_per_liter"])) * annual_litres, 2
            ),
            spread_z=float(basis["spread_z"]),
            spread_percentile=float(basis["spread_percentile"]),
            history_weeks=int(basis["history_weeks"]),
            window_weeks=int(basis["window_weeks"]),
            disclaimer=(
                "Diferenca medida entre duas series publicas da ANP, revenda media. "
                "Nao e cotacao de fornecedor e nao considera frete, contrato ou tributo "
                "especifico do comprador."
            ),
        )

    # ------------------------------------------------------------- challenger

    def _challenger_view(self, origin_price: float, target_date: str) -> ModelView | None:
        """Le a previsao do challenger, se ela for da mesma semana-alvo.

        Comparar previsoes de semanas diferentes produziria uma divergencia
        inventada, entao previsao desalinhada e simplesmente descartada.
        """

        if not self.challenger:
            return None
        forecast = self.challenger.get("forecast")
        if not isinstance(forecast, dict):
            return None
        if str(forecast.get("target_date")) != target_date:
            return None
        point = float(forecast["point"])
        change = point - origin_price
        return ModelView(
            name="paridade",
            role="challenger",
            point_brl_per_liter=point,
            expected_change_brl_per_liter=change,
            direction=_direction(change, self.signal_threshold),
            would_trigger=bool(change > self.signal_threshold),
        )

    # ---------------------------------------------------------------- decisao

    def _size(self, volume: float, change: float, triggers: bool) -> tuple[float, float]:
        weekly_liters = volume * 12.0 / 52.0
        liters = weekly_liters * self.flexibility if triggers else 0.0
        return liters, round(liters * max(change, 0.0), 2)

    def decide_state(self, uf: str, volume_liters: float = 200_000.0) -> S10Decision:
        """Decisao para o preco que o comprador daquele estado efetivamente paga.

        A evidencia do modelo estadual e apenas de desenvolvimento, e isso limita
        a confianca por construcao: nenhuma decisao estadual sai como ``alta``
        enquanto o holdout do estado nao for aberto e as semanas prospectivas nao
        se acumularem.
        """

        volume = float(volume_liters)
        if not 0 < volume <= 50_000_000:
            raise ValueError("volume_liters precisa ser positivo e no maximo 50 milhoes")
        status = self.product.status()
        if not status.serving_ready:
            raise RuntimeError("a release nao esta apta a servir; decisao bloqueada")

        payload = self._regional_payload(uf)
        code = str(payload["uf"]).strip().upper()
        forecast = payload["forecast"]
        origin_price = float(forecast["origin_price"])
        change = float(forecast["point"]) - origin_price
        triggers = bool(change > self.signal_threshold)
        view = ModelView(
            name=f"regional_{code.lower()}",
            role="primary",
            point_brl_per_liter=float(forecast["point"]),
            expected_change_brl_per_liter=change,
            direction=_direction(change, self.signal_threshold),
            would_trigger=triggers,
        )
        liters, saving = self._size(volume, change, triggers)
        evidence = payload.get("evidence") or {}
        evidence_status = str(evidence.get("status", "unknown"))
        settled = int(evidence.get("prospective_weeks_settled", 0))
        target = int(evidence.get("prospective_target", 26))

        rationale = [
            f"Preco {code} de R$ {origin_price:.4f}/L, contra R$ "
            f"{float(forecast['national_point']):.4f}/L da media nacional prevista: "
            f"a decisao usa a serie que o comprador do estado enfrenta.",
            f"Decomposicao: nacional R$ {float(forecast['national_point']):.4f}/L "
            f"mais spread R$ {float(forecast['spread_point']):+.4f}/L.",
        ]
        if triggers:
            rationale.append(
                f"Alta prevista de R$ {change:.4f}/L supera o limiar de R$ "
                f"{self.signal_threshold:.2f}/L."
            )
        else:
            rationale.append(
                f"Variacao prevista de R$ {change:+.4f}/L nao supera o limiar de R$ "
                f"{self.signal_threshold:.2f}/L: antecipar nao se paga."
            )
        if evidence_status != "holdout_read":
            confidence = "media" if abs(change) >= 2.0 * self.signal_threshold else "baixa"
            rationale.append(
                f"Evidencia do modelo estadual: {evidence_status}, com {settled}/{target} "
                "semanas prospectivas liquidadas. A confianca nao sobe acima de media "
                "enquanto o holdout do estado nao for aberto."
            )
        else:  # pragma: no cover - reservado para quando o holdout for aberto
            confidence = "alta" if abs(change) >= 2.0 * self.signal_threshold else "media"

        return S10Decision(
            target_date=str(forecast["target_date"]),
            origin_date=str(forecast["origin_date"]),
            origin_price_brl_per_liter=origin_price,
            volume_liters=volume,
            recommendation="antecipar" if triggers else "aguardar",
            liters_to_prebuy=round(liters, 1),
            expected_saving_brl=saving,
            exposure_if_wrong_brl=round(liters * abs(change), 2),
            signal_threshold_brl_per_liter=self.signal_threshold,
            confidence=confidence,
            models_agree=True,
            decided_by=view.name,
            scope=f"estadual:{code}",
            evidence_status=evidence_status,
            views=(view,),
            interval_low_brl_per_liter=float(forecast["lower"]),
            interval_high_brl_per_liter=float(forecast["upper"]),
            interval_nominal_coverage=float(forecast.get("nominal_coverage", 0.80)),
            rationale=tuple(rationale),
            disclaimer=self.DISCLAIMER,
        )

    def decide(
        self, volume_liters: float = 200_000.0, *, uf: str | None = None
    ) -> S10Decision:
        if uf:
            return self.decide_state(uf, volume_liters)
        volume = float(volume_liters)
        if not 0 < volume <= 50_000_000:
            raise ValueError("volume_liters precisa ser positivo e no maximo 50 milhoes")
        status = self.product.status()
        if not status.serving_ready:
            raise RuntimeError("a release nao esta apta a servir; decisao bloqueada")

        forecast = self.product.current_forecast
        origin_price = float(forecast.last_observed_price)
        change = float(forecast.point) - origin_price
        primary = ModelView(
            name=forecast.primary_model,
            role="primary",
            point_brl_per_liter=float(forecast.point),
            expected_change_brl_per_liter=change,
            direction=_direction(change, self.signal_threshold),
            would_trigger=bool(change > self.signal_threshold),
        )
        challenger = self._challenger_view(origin_price, forecast.target_date)
        views = (primary,) if challenger is None else (primary, challenger)

        agree = challenger is None or challenger.direction == primary.direction
        liters, _ = self._size(volume, change, primary.would_trigger)
        recommendation = "antecipar" if primary.would_trigger else "aguardar"

        rationale: list[str] = []
        if primary.would_trigger:
            rationale.append(
                f"{primary.name} preve alta de R$ {change:.4f}/L, acima do limiar "
                f"pre-registrado de R$ {self.signal_threshold:.2f}/L."
            )
        else:
            rationale.append(
                f"{primary.name} preve variacao de R$ {change:+.4f}/L, abaixo do limiar "
                f"de R$ {self.signal_threshold:.2f}/L: antecipar nao se paga."
            )
        if challenger is not None:
            if agree:
                rationale.append(
                    f"O challenger paridade concorda na direcao ({challenger.direction}), "
                    f"com R$ {challenger.point_brl_per_liter:.4f}/L."
                )
            else:
                rationale.append(
                    f"O challenger paridade discorda: preve {challenger.direction} para "
                    f"R$ {challenger.point_brl_per_liter:.4f}/L. A recomendacao segue o "
                    "modelo primario da release, mas a confianca cai."
                )
        else:
            rationale.append(
                "Sem previsao de challenger alinhada a esta semana-alvo; decisao "
                "apoiada apenas no modelo primario."
            )

        if not agree:
            confidence = "baixa"
        elif abs(change) >= 2.0 * self.signal_threshold:
            confidence = "alta"
        else:
            confidence = "media"
            rationale.append(
                "Sinal proximo do limiar: pequenas revisoes de dado podem inverter a "
                "recomendacao."
            )

        return S10Decision(
            target_date=forecast.target_date,
            origin_date=forecast.last_observed_date,
            origin_price_brl_per_liter=origin_price,
            volume_liters=volume,
            recommendation=recommendation,
            liters_to_prebuy=round(liters, 1),
            expected_saving_brl=round(liters * max(change, 0.0), 2),
            # Se a alta nao vier, o custo e ter comprado caro o mesmo volume.
            exposure_if_wrong_brl=round(liters * abs(change), 2),
            signal_threshold_brl_per_liter=self.signal_threshold,
            confidence=confidence,
            models_agree=bool(agree),
            decided_by=primary.name,
            scope="nacional",
            evidence_status="holdout_read",
            views=views,
            interval_low_brl_per_liter=float(forecast.p10),
            interval_high_brl_per_liter=float(forecast.p90),
            interval_nominal_coverage=0.80,
            rationale=tuple(rationale),
            disclaimer=self.DISCLAIMER,
        )

    # ------------------------------------------------------------- governanca

    def governance(self) -> dict[str, Any]:
        """Por que o primario e o primario, com os gates que decidiram isso."""

        payload: dict[str, Any] = {
            "contract_version": "1.0",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "primary_model": self.product.current_forecast.primary_model,
            "states_available": list(self.available_states),
            "prospective": {
                name: status.as_dict()
                for name, status in review_ledgers(self.ledger_paths).items()
            },
            "promotion_policy": (
                "challenger so vira primario com todos os gates decidiveis aprovados e "
                "confirmacao prospectiva; promocao automatica e proibida"
            ),
        }
        if not self.gate_manifest:
            payload["gate_review"] = None
            return payload
        reviews = {
            "paridade_vs_arima": self.gate_manifest.get("verdict_parity_vs_arima"),
            "vs_epl_krls_vs_arima": self.gate_manifest.get("verdict_vs_epl_krls_vs_arima"),
        }
        payload["gate_review"] = {
            "holdout_window": self.gate_manifest.get("holdout_window"),
            "verdicts": {
                name: {
                    "promote": verdict.get("promote"),
                    "failed_gates": verdict.get("failed_gates"),
                    "gates": verdict.get("gates"),
                }
                for name, verdict in reviews.items()
                if isinstance(verdict, dict)
            },
            "interval_review": self.gate_manifest.get("interval_review", {}).get(
                "adaptive_conformal"
            ),
        }
        return payload
