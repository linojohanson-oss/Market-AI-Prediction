import sys
from pathlib import Path
from datetime import datetime
import warnings

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import yfinance as yf

from config import MarketConfig


CONTEXT_VARIABLES_DEFAULT = {
    "^GSPC": "SP500",
    "^IXIC": "NASDAQ",
    "^DJI": "DOWJONES",
    "^VIX": "VIX",
    "DX-Y.NYB": "DXY",
    "GC=F": "GOLD",
    "CL=F": "OIL_WTI",
    "BZ=F": "OIL_BRENT",
    "BTC-USD": "BTC",
    "ETH-USD": "ETH",
    "ARS=X": "USD_ARS",
    "BRL=X": "USD_BRL",
}


def normalizar_columnas(df):
    if isinstance(df.columns, pd.MultiIndex):
        nuevas = []
        for col in df.columns:
            partes = [str(x).strip() for x in col if str(x).strip()]
            nuevas.append(partes[0] if partes else "columna")
        df.columns = nuevas

    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def calcular_rsi(close, window=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def descargar_contexto_ticker(ticker, nombre, period="10y", interval="1d"):
    inicio = datetime.now()

    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            group_by="column",
            threads=False,
        )

        if df is None or df.empty:
            return None, f"{ticker}: sin datos"

        df = normalizar_columnas(df)
        df.reset_index(inplace=True)

        if "Datetime" in df.columns and "Date" not in df.columns:
            df.rename(columns={"Datetime": "Date"}, inplace=True)

        if "Date" not in df.columns or "Close" not in df.columns:
            return None, f"{ticker}: faltan Date/Close"

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

        df = df[["Date", "Close"]].copy()
        df.dropna(subset=["Date", "Close"], inplace=True)
        df.sort_values("Date", inplace=True)
        df.drop_duplicates(subset=["Date"], keep="last", inplace=True)

        df.rename(columns={"Close": nombre}, inplace=True)

        return df, ""

    except Exception as e:
        return None, f"{ticker}: {e}"


def calcular_features_contexto(df, nombre):
    df = df.sort_values("Date").copy()

    precio = df[nombre]

    df[f"{nombre}_Return_1d"] = precio.pct_change(1)
    df[f"{nombre}_Return_5d"] = precio.pct_change(5)
    df[f"{nombre}_Return_10d"] = precio.pct_change(10)
    df[f"{nombre}_Return_20d"] = precio.pct_change(20)

    df[f"{nombre}_EMA_20"] = precio.ewm(span=20, adjust=False).mean()
    df[f"{nombre}_EMA_50"] = precio.ewm(span=50, adjust=False).mean()
    df[f"{nombre}_EMA_200"] = precio.ewm(span=200, adjust=False).mean()

    df[f"{nombre}_RSI_14"] = calcular_rsi(precio, 14)
    df[f"{nombre}_Volatility_20"] = df[f"{nombre}_Return_1d"].rolling(20).std()
    df[f"{nombre}_Momentum_20"] = precio - precio.shift(20)

    df[f"{nombre}_vs_EMA20"] = precio / df[f"{nombre}_EMA_20"] - 1
    df[f"{nombre}_vs_EMA50"] = precio / df[f"{nombre}_EMA_50"] - 1
    df[f"{nombre}_vs_EMA200"] = precio / df[f"{nombre}_EMA_200"] - 1

    return df


def generar_contexto_mercado(nombre_mercado="merval"):
    inicio = datetime.now()
    market = MarketConfig(nombre_mercado)

    print("=" * 70)
    print(f"MARKET CONTEXT ENGINE - {market.display_name}")
    print("=" * 70)
    print(f"Mercado      : {market.market}")
    print(f"Periodo      : {market.period}")
    print(f"Intervalo    : {market.interval}")

    variables = CONTEXT_VARIABLES_DEFAULT.copy()

    print(f"Variables contexto: {len(variables)}")
    print("=" * 70)

    datasets = []
    errores = []

    for ticker, nombre in variables.items():
        df, error = descargar_contexto_ticker(
            ticker=ticker,
            nombre=nombre,
            period=market.period,
            interval=market.interval,
        )

        if df is None:
            errores.append(error)
            print(f"ERROR {ticker:12} {error}")
            continue

        df = calcular_features_contexto(df, nombre)
        datasets.append(df)

        print(f"OK    {ticker:12} -> {nombre:12} {len(df):6} registros")

    if not datasets:
        raise RuntimeError("No se pudo descargar ninguna variable de contexto.")

    contexto = datasets[0]

    for df in datasets[1:]:
        contexto = contexto.merge(df, on="Date", how="outer")

    contexto.sort_values("Date", inplace=True)
    contexto.ffill(inplace=True)
    contexto.bfill(inplace=True)

    contexto.replace([np.inf, -np.inf], np.nan, inplace=True)
    contexto.fillna(0, inplace=True)

    archivo_csv = market.processed_dir / f"{market.market.upper()}_CONTEXTO.csv"
    archivo_excel = market.processed_dir / f"{market.market.upper()}_CONTEXTO.xlsx"

    contexto.to_csv(archivo_csv, index=False, encoding="utf-8-sig")
    contexto.to_excel(archivo_excel, index=False)

    print()
    print("=" * 70)
    print(f"RESUMEN CONTEXTO - {market.display_name}")
    print("=" * 70)
    print(f"Variables descargadas : {len(datasets)}")
    print(f"Errores               : {len(errores)}")
    print(f"Registros             : {len(contexto):,}")
    print(f"Columnas              : {len(contexto.columns)}")
    print(f"Desde                 : {contexto['Date'].min()}")
    print(f"Hasta                 : {contexto['Date'].max()}")
    print(f"CSV generado          : {archivo_csv}")
    print(f"Excel generado        : {archivo_excel}")
    print(f"Tiempo total          : {datetime.now() - inicio}")

    if errores:
        print()
        print("Errores detectados")
        print("-" * 70)
        for e in errores:
            print(e)

    return contexto


if __name__ == "__main__":
    generar_contexto_mercado("merval")