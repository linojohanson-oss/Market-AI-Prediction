from types import SimpleNamespace

import pandas as pd

from config import MarketConfig
from source.market_dataset import cargar_indicadores
from source.market_indicators import calcular_indicadores_activo


def test_market_config_incluye_timestamp_en_nombres_de_archivo():
    market = MarketConfig("crypto")

    assert market.full_raw_csv.name == "CRYPTO_COMPLETO.csv"
    assert market.full_raw_excel.name == "CRYPTO_COMPLETO.xlsx"
    assert market.indicators_csv.name.startswith("CRYPTO_INDICADORES_")
    assert market.dataset_csv.name.startswith("CRYPTO_DATASET_IA_")
    assert market.execution_timestamp.count("_") == 1


def test_cargar_indicadores_usa_el_archivo_mas_reciente(tmp_path):
    market = SimpleNamespace(
        market="crypto",
        processed_dir=tmp_path,
    )

    older = tmp_path / "CRYPTO_INDICADORES_20240101_010101.csv"
    latest = tmp_path / "CRYPTO_INDICADORES_20240102_010101.csv"

    pd.DataFrame(
        {
            "Date": ["2024-01-02"],
            "Ticker": ["BTC-USD"],
            "Open": [10],
            "High": [11],
            "Low": [9],
            "Close": [10],
            "Volume": [100],
        }
    ).to_csv(latest, index=False)

    pd.DataFrame(
        {
            "Date": ["2024-01-01"],
            "Ticker": ["BTC-USD"],
            "Open": [9],
            "High": [10],
            "Low": [8],
            "Close": [9],
            "Volume": [100],
        }
    ).to_csv(older, index=False)

    df = cargar_indicadores(market)

    assert df.iloc[0]["Date"] == pd.Timestamp("2024-01-02")


def test_calcular_indicadores_activo_incluye_patrones_candlestick():
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "Open": [10.0, 10.0, 11.0, 10.0],
            "High": [11.0, 11.2, 11.5, 12.5],
            "Low": [9.0, 8.0, 9.8, 9.8],
            "Close": [10.0, 10.8, 10.2, 12.0],
            "Volume": [100, 100, 100, 100],
        }
    )

    resultado = calcular_indicadores_activo(df)

    assert "ADX_14" in resultado.columns
    assert "OBV" in resultado.columns
    assert "VWAP" in resultado.columns
    assert "Stoch_K" in resultado.columns
    assert "Stoch_D" in resultado.columns
    assert "CCI" in resultado.columns
    assert "Williams_R" in resultado.columns
    assert "Keltner_Upper" in resultado.columns
    assert "Keltner_Lower" in resultado.columns
    assert "CMF" in resultado.columns
    assert "Volatility_Std_20" in resultado.columns
    assert "Candle_Doji" in resultado.columns
    assert "Candle_Hammer" in resultado.columns
    assert "Candle_Hanging_Man" in resultado.columns
    assert "Candle_Shooting_Star" in resultado.columns
    assert "Candle_Bearish_Belt_Hold" in resultado.columns
    assert "Candle_Bullish_Engulfing" in resultado.columns
    assert "Candle_Bearish_Engulfing" in resultado.columns
    assert "Candle_Tweezer_Top" in resultado.columns
    assert "Candle_Tweezer_Bottom" in resultado.columns
    assert "Candle_Piercing" in resultado.columns
    assert "Candle_Bullish_Separating_Line" in resultado.columns
    assert "Candle_Bullish_Kicker" in resultado.columns
    assert "Candle_Bearish_Kicker" in resultado.columns
    assert "Candle_Dark_Cloud_Cover" in resultado.columns
    assert "Candle_Morning_Star" in resultado.columns
    assert "Candle_Evening_Star" in resultado.columns
    assert "Candle_Evening_Doji_Star" in resultado.columns
    assert "Candle_Three_White_Soldiers" in resultado.columns
    assert "Candle_Three_Black_Crows" in resultado.columns
    assert "Candle_Three_White_Strike" in resultado.columns
    assert "Candle_Inside_Bar" in resultado.columns
    assert "Candle_Bullish_Abandoned_Baby" in resultado.columns
    assert "Candle_Bearish_Abandoned_Baby" in resultado.columns
    assert "Candle_Three_Inside_Up" in resultado.columns
    assert "Candle_Three_Outside_Up" in resultado.columns
    assert "Candle_Upside_Gap_Two_Crows" in resultado.columns
    assert "Candle_Bullish_Harami" in resultado.columns
    assert "Candle_Bearish_Harami" in resultado.columns
    assert "Candle_Spinning_Top" in resultado.columns
    assert "Candle_Concealing_Baby_Swallow" in resultado.columns
    assert "Candle_Rising_Three_Methods" in resultado.columns
    assert "Candle_Mat_Hold" in resultado.columns
    assert "Candle_Ladder_Bottom" in resultado.columns
    assert resultado["Candle_Doji"].iloc[0] == 1
    assert resultado["Candle_Hammer"].iloc[1] == 1
    assert resultado["Candle_Bullish_Engulfing"].iloc[3] == 1
