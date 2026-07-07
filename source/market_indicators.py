import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import pandas as pd
import numpy as np

from config import MarketConfig


TARGETS = [1, 5, 10, 20, 60]


def calcular_rsi(close, window=14):
    delta = close.diff()

    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()

    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calcular_atr(df, window=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()

    return atr


def calcular_indicadores_activo(df):
    df = df.sort_values("Date").copy()

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
    df["High"] = pd.to_numeric(df["High"], errors="coerce")
    df["Low"] = pd.to_numeric(df["Low"], errors="coerce")

    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    else:
        df["Volume"] = 0

    # Retornos
    df["Return_1d"] = df["Close"].pct_change(1)
    df["Return_5d"] = df["Close"].pct_change(5)
    df["Return_10d"] = df["Close"].pct_change(10)
    df["Return_20d"] = df["Close"].pct_change(20)

    # Medias simples
    for w in [5, 10, 20, 50, 100, 200]:
        df[f"SMA_{w}"] = df["Close"].rolling(w).mean()

    # Medias exponenciales
    for w in [12, 20, 26, 50, 100, 200]:
        df[f"EMA_{w}"] = df["Close"].ewm(span=w, adjust=False).mean()

    # Momentum
    df["RSI_14"] = calcular_rsi(df["Close"], 14)
    df["ROC_10"] = df["Close"].pct_change(10)
    df["Momentum_10"] = df["Close"] - df["Close"].shift(10)

    # MACD
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # Bollinger Bands
    sma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()

    df["BB_Upper"] = sma20 + 2 * std20
    df["BB_Lower"] = sma20 - 2 * std20
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / sma20

    # Volatilidad
    df["ATR_14"] = calcular_atr(df, 14)
    df["Volatility_20"] = df["Return_1d"].rolling(20).std()

    # Volumen
    df["Volume_SMA20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA20"].replace(0, np.nan)

    # Tendencia / posición relativa
    df["Close_vs_SMA20"] = df["Close"] / df["SMA_20"] - 1
    df["Close_vs_SMA50"] = df["Close"] / df["SMA_50"] - 1
    df["Close_vs_SMA200"] = df["Close"] / df["SMA_200"] - 1

    df["Trend_SMA20_50"] = np.where(df["SMA_20"] > df["SMA_50"], 1, 0)
    df["Trend_SMA50_200"] = np.where(df["SMA_50"] > df["SMA_200"], 1, 0)

    # Targets multihorizonte
    for t in TARGETS:
        df[f"Future_Close_{t}d"] = df["Close"].shift(-t)
        df[f"Future_Return_{t}d"] = df[f"Future_Close_{t}d"] / df["Close"] - 1
        df[f"Target_{t}d"] = np.where(df[f"Future_Return_{t}d"] > 0, 1, 0)

    return df


def generar_indicadores(nombre_mercado="merval"):
    inicio = datetime.now()
    market = MarketConfig(nombre_mercado)

    print("=" * 70)
    print(f"MARKET INDICATORS - {market.display_name}")
    print("=" * 70)

    if not market.full_raw_csv.exists():
        raise FileNotFoundError(
            f"No existe archivo base: {market.full_raw_csv}. "
            f"Primero ejecutá market_downloader.py"
        )

    df = pd.read_csv(market.full_raw_csv)
    df["Date"] = pd.to_datetime(df["Date"])

    print(f"Mercado        : {market.market}")
    print(f"Archivo origen : {market.full_raw_csv}")
    print(f"Registros base : {len(df):,}")
    print(f"Activos        : {df['Ticker'].nunique()}")

    resultados = []

    for ticker, df_ticker in df.groupby("Ticker"):
        print(f"Procesando {ticker}...")
        procesado = calcular_indicadores_activo(df_ticker)
        resultados.append(procesado)

    base = pd.concat(resultados, ignore_index=True)
    base.sort_values(["Ticker", "Date"], inplace=True)

    base.to_csv(market.indicators_csv, index=False, encoding="utf-8-sig")
    base.to_excel(market.indicators_excel, index=False)

    columnas_indicadores = [
        c for c in base.columns
        if c.startswith((
            "Return_",
            "SMA_",
            "EMA_",
            "RSI_",
            "ROC_",
            "Momentum_",
            "MACD",
            "BB_",
            "ATR_",
            "Volatility_",
            "Volume_",
            "Close_vs_",
            "Trend_",
        ))
    ]

    columnas_targets = [c for c in base.columns if c.startswith("Target_")]

    print()
    print("=" * 70)
    print(f"RESUMEN INDICADORES - {market.display_name}")
    print("=" * 70)
    print(f"Activos procesados      : {base['Ticker'].nunique()}")
    print(f"Registros totales       : {len(base):,}")
    print(f"Indicadores generados   : {len(columnas_indicadores)}")
    print(f"Targets generados       : {len(columnas_targets)}")
    print(f"Columnas finales        : {len(base.columns)}")
    print(f"Desde                   : {base['Date'].min()}")
    print(f"Hasta                   : {base['Date'].max()}")
    print(f"CSV generado            : {market.indicators_csv}")
    print(f"Excel generado          : {market.indicators_excel}")
    print(f"Tiempo total            : {datetime.now() - inicio}")

    return base


if __name__ == "__main__":
    generar_indicadores("merval")