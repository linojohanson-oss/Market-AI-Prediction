from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import source.market_dataset as market_dataset


def test_generar_dataset_ia_normaliza_ticker_en_minusculas(tmp_path, monkeypatch):
    indicadores_path = tmp_path / "TEST_INDICADORES.csv"
    dataset_path = tmp_path / "TEST_DATASET_IA.csv"
    excel_path = tmp_path / "TEST_DATASET_IA.xlsx"

    pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "Open": [10, 11, 12],
            "High": [11, 12, 13],
            "Low": [9, 10, 11],
            "Close": [10, 11, 12],
            "Volume": [100, 101, 102],
            "Target_5d": [1, 0, 1],
        }
    ).to_csv(indicadores_path, index=False)

    market = SimpleNamespace(
        market="test",
        display_name="TEST",
        indicators_csv=indicadores_path,
        dataset_csv=dataset_path,
        dataset_excel=excel_path,
        processed_dir=tmp_path,
    )

    monkeypatch.setattr(market_dataset, "MarketConfig", lambda _: market)
    monkeypatch.setattr(pd.DataFrame, "to_excel", lambda self, *args, **kwargs: None)

    dataset = market_dataset.generar_dataset_ia("test")

    assert "Ticker" in dataset.columns
    assert dataset["Ticker"].eq("AAPL").all()
    assert dataset["Objetivo_5d"].notna().all()
    assert dataset_path.exists()
