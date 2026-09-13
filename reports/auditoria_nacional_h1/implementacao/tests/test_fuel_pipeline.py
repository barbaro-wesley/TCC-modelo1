from pathlib import Path

import pandas as pd
import pytest

from vs_epl_krls import (
    VSEPLKRLS,
    evaluate_fuel_product,
    load_anp_fuel_csv,
    make_lagged_dataset,
)


def write_fixture(path: Path, n: int = 45):
    rows = []
    dates = pd.date_range("2019-01-06", periods=n, freq="W-SUN")
    for product, offset in [("OLEO DIESEL S10", 0.0), ("ÓLEO DIESEL S-500", 0.4)]:
        for index, date in enumerate(dates):
            rows.append(
                {
                    "Data": date.strftime("%d/%m/%Y"),
                    "Produto": product,
                    "Preço Médio": f"{4.0 + offset + 0.01 * index:.2f}".replace(".", ","),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False, sep=";")


def test_load_anp_csv_identifies_columns_products_and_decimal_comma(tmp_path):
    path = tmp_path / "anp.csv"
    write_fixture(path)
    frame = load_anp_fuel_csv(path)
    assert set(frame["product"]) == {"S10", "S500"}
    assert frame["price"].dtype.kind == "f"
    assert frame.groupby("product").size().to_dict() == {"S10": 45, "S500": 45}


def test_lags_and_targets_are_temporally_aligned(tmp_path):
    path = tmp_path / "anp.csv"
    write_fixture(path)
    frame = load_anp_fuel_csv(path)
    lagged = make_lagged_dataset(frame, lags=[0, 1, 3], horizon=2)
    row = lagged[lagged["product"] == "S10"].iloc[0]
    source = frame[frame["product"] == "S10"].reset_index(drop=True)
    origin = source.index[source["date"] == row["date"]][0]
    assert row["lag_0"] == pytest.approx(source.loc[origin, "price"])
    assert row["lag_3"] == pytest.approx(source.loc[origin - 3, "price"])
    assert row["target"] == pytest.approx(source.loc[origin + 2, "price"])
    assert row["target_date"] > row["date"]


def test_delayed_prequential_fuel_evaluation(tmp_path):
    path = tmp_path / "anp.csv"
    write_fixture(path, n=60)
    frame = load_anp_fuel_csv(path)
    lagged = make_lagged_dataset(frame, lags=[0, 1, 2], horizon=2)

    def factory():
        return VSEPLKRLS(
            enable_rule_merging=False,
            adapt_kernel_width=False,
            max_dictionary_size=8,
        )

    result = evaluate_fuel_product(
        lagged,
        product="S10",
        horizon=2,
        train_fraction=0.75,
        model_factory=factory,
    )
    assert result.metrics["rmse"] >= 0
    assert len(result.predictions) > 0
    assert (result.predictions["target_date"] > result.predictions["origin_date"]).all()
    assert result.model_summary["n_seen"] < len(lagged[lagged["product"] == "S10"])


def test_csv_loader_fails_loudly_instead_of_inventing_data(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_anp_fuel_csv(tmp_path / "missing.csv")

    bad = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="identify"):
        load_anp_fuel_csv(bad)


def test_generic_diesel_requires_explicit_s500_mapping(tmp_path):
    path = tmp_path / "generic.csv"
    pd.DataFrame(
        {
            "Data": ["01/01/2020"],
            "Produto": ["ÓLEO DIESEL"],
            "Preço": ["4,50"],
        }
    ).to_csv(path, index=False, sep=";")
    with pytest.raises(ValueError, match="no valid"):
        load_anp_fuel_csv(path, products=["S500"])
    mapped = load_anp_fuel_csv(
        path,
        products=["S500"],
        generic_diesel_as_s500=True,
    )
    assert mapped.iloc[0]["product"] == "S500"


def test_official_style_excel_is_supported(tmp_path):
    path = tmp_path / "anp.xlsx"
    pd.DataFrame(
        {
            "DATA INICIAL": [pd.Timestamp("2020-01-05"), pd.Timestamp("2020-01-12")],
            "DATA FINAL": [pd.Timestamp("2020-01-11"), pd.Timestamp("2020-01-18")],
            "PRODUTO": ["OLEO DIESEL S10", "OLEO DIESEL S10"],
            "PRECO MEDIO REVENDA": [4.1, 4.2],
        }
    ).to_excel(path, index=False)
    loaded = load_anp_fuel_csv(path, products=["S10"])
    assert loaded["date"].tolist() == [pd.Timestamp("2020-01-05"), pd.Timestamp("2020-01-12")]
    assert loaded["price"].tolist() == [4.1, 4.2]
