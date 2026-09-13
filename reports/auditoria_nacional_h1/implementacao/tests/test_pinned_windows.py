"""A janela congelada do S10 nao pode se mover quando chega dado novo.

Ate 2026-08-24 o protocolo usava ``development_end = n - holdout_size``: o corte
era ancorado no fim da serie, entao cada semana nova publicada pela ANP deslocava
os tres folds e o holdout.  Duas consequencias medidas: os numeros ja divulgados
deixavam de ser reproduziveis, e reexecutar a selecao lia um holdout diferente do
que ja tinha sido lido, sem nenhum aviso.  Estes testes fixam o contrato novo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vs_epl_krls.selection import (
    S10_HOLDOUT_END,
    S10_HOLDOUT_SIZE,
    S10_HOLDOUT_START,
    expanding_validation_folds,
    pinned_validation_folds,
)

#: Tamanho do painel que gerou a evidencia publicada do modelo de paridade.
PUBLISHED_WEEKS = 702
#: Grade semanal sintetica alinhada ao fim do holdout congelado, para que os
#: indices esperados aqui sejam os mesmos do manifesto publicado sem depender de
#: nenhum arquivo baixado.
PANEL_START = pd.date_range(end=S10_HOLDOUT_END, periods=PUBLISHED_WEEKS, freq="7D")[0]


def weekly_dates(n: int, start: pd.Timestamp | str = PANEL_START) -> pd.Series:
    return pd.Series(pd.date_range(start, periods=n, freq="7D"))


def test_pinned_window_reproduces_the_published_evidence_indices() -> None:
    windows = pinned_validation_folds(weekly_dates(PUBLISHED_WEEKS))

    assert [
        (fold.validation_start, fold.validation_end) for fold in windows.folds
    ] == [(442, 494), (494, 546), (546, 598)]
    assert (windows.holdout.validation_start, windows.holdout.validation_end) == (598, 702)
    assert windows.development_end == 598
    assert windows.prospective.validation_end == windows.prospective.validation_start


def test_pinned_window_does_not_move_when_the_panel_grows() -> None:
    reference = pinned_validation_folds(weekly_dates(PUBLISHED_WEEKS))

    for extra in (1, 5, 52):
        grown = pinned_validation_folds(weekly_dates(PUBLISHED_WEEKS + extra))
        assert grown.folds == reference.folds
        assert grown.holdout == reference.holdout
        assert grown.development_end == reference.development_end
        # Todo dado novo vira cauda prospectiva, fora de desenvolvimento e holdout.
        assert grown.prospective.validation_start == reference.holdout.validation_end
        assert (
            grown.prospective.validation_end - grown.prospective.validation_start == extra
        )


def test_expanding_folds_still_slide_which_is_why_the_pinned_window_exists() -> None:
    published, _ = expanding_validation_folds(PUBLISHED_WEEKS)
    grown, _ = expanding_validation_folds(PUBLISHED_WEEKS + 1)

    assert grown[0].validation_start == published[0].validation_start + 1
    assert grown[-1].validation_end == published[-1].validation_end + 1


def test_pinned_window_covers_exactly_the_frozen_holdout_dates() -> None:
    dates = weekly_dates(PUBLISHED_WEEKS)
    windows = pinned_validation_folds(dates)
    holdout = dates[windows.holdout.validation_start : windows.holdout.validation_end]

    assert len(holdout) == S10_HOLDOUT_SIZE
    assert str(holdout.iloc[0].date()) == S10_HOLDOUT_START
    assert str(holdout.iloc[-1].date()) == S10_HOLDOUT_END


def test_missing_pinned_date_is_a_loud_failure() -> None:
    deslocada = PANEL_START + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="inicio do holdout"):
        pinned_validation_folds(weekly_dates(PUBLISHED_WEEKS, start=deslocada))

    truncated = weekly_dates(PUBLISHED_WEEKS).iloc[:600]
    with pytest.raises(ValueError, match="fim do holdout"):
        pinned_validation_folds(truncated)


def test_gap_in_the_series_is_a_loud_failure() -> None:
    dates = weekly_dates(PUBLISHED_WEEKS).drop(index=600).reset_index(drop=True)

    with pytest.raises(ValueError, match="esperado 104"):
        pinned_validation_folds(dates)


def test_unsorted_and_short_histories_are_rejected() -> None:
    with pytest.raises(ValueError, match="ordem crescente"):
        pinned_validation_folds(weekly_dates(PUBLISHED_WEEKS).iloc[::-1])

    with pytest.raises(ValueError, match="historico insuficiente"):
        pinned_validation_folds(
            weekly_dates(PUBLISHED_WEEKS), validation_size=52, n_folds=12
        )


def test_manifest_records_the_pin_for_auditing() -> None:
    manifest = pinned_validation_folds(weekly_dates(PUBLISHED_WEEKS + 3)).as_manifest()

    assert manifest["holdout_start_date"] == S10_HOLDOUT_START
    assert manifest["holdout_end_date"] == S10_HOLDOUT_END
    assert manifest["development_end_index"] == 598
    assert manifest["prospective_weeks"] == 3


def test_supervised_target_dates_are_accepted_as_numpy_datetimes() -> None:
    dates = weekly_dates(PUBLISHED_WEEKS).to_numpy().astype("datetime64[ns]")

    windows = pinned_validation_folds(dates)

    assert windows.development_end == 598
    assert isinstance(windows.holdout.validation_start, int)
    assert np.all(np.diff(dates).astype("timedelta64[D]").astype(int) == 7)


def test_gaps_before_the_holdout_do_not_break_the_pin() -> None:
    """A serie semanal real da ANP tem semanas faltando; o corte por data lida
    com isso, o corte por deslocamento nao."""

    dates = weekly_dates(PUBLISHED_WEEKS + 9)
    with_gaps = dates.drop(index=[10, 40, 90, 150, 200, 260, 310, 380, 420])
    with_gaps = with_gaps.reset_index(drop=True)

    windows = pinned_validation_folds(with_gaps)

    assert windows.holdout.validation_end - windows.holdout.validation_start == 104
    assert str(with_gaps[windows.holdout.validation_start].date()) == S10_HOLDOUT_START
    assert str(with_gaps[windows.holdout.validation_end - 1].date()) == S10_HOLDOUT_END
