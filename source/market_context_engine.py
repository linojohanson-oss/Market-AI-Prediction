import sys
import os
from pathlib import Path
from datetime import datetime
import warnings

ROOT = Path(__file__).resolve().parent.parent

from source.app_logger import log_print as print

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import yfinance as yf
from pandas.api.types import is_datetime64tz_dtype

from config import MarketConfig
from config import get_json_config_value
from source.market_sqlite_cache import MarketSQLiteCache


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

SQLITE_INCREMENTAL_CACHE = bool(get_json_config_value("pipeline.sqlite_incremental_cache_enabled", True))
CONTEXT_RECALC_LOOKBACK_BARS = int(get_json_config_value("pipeline.context_recalc_lookback_bars", 320))


def _parse_bool_env(value, default=False):
    if value is None:
        return default

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def exportar_excel_contexto_habilitado():
    global_default = _parse_bool_env(os.getenv("MARKET_EXPORT_EXCEL"), default=False)
    return _parse_bool_env(os.getenv("MARKET_CONTEXT_EXPORT_EXCEL"), default=global_default)


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


def quitar_timezone_para_excel(df):
    df = df.copy()

    for col in df.columns:
        if is_datetime64tz_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    return df


def intervalo_a_lookback_timedelta(interval, lookback_bars):
    text = str(interval).strip().lower()

    if text.endswith("h"):
        n = int(text[:-1]) if text[:-1].isdigit() else 1
        return pd.Timedelta(hours=n * lookback_bars)

    if text.endswith("d"):
        n = int(text[:-1]) if text[:-1].isdigit() else 1
        return pd.Timedelta(days=n * lookback_bars)

    if text.endswith("wk"):
        n = int(text[:-2]) if text[:-2].isdigit() else 1
        return pd.Timedelta(days=7 * n * lookback_bars)

    return pd.Timedelta(days=lookback_bars)


def descargar_contexto_ticker(ticker, nombre, period="10y", interval="1d", start_date=None):
    inicio = datetime.now()

    try:
        kwargs = {
            "interval": interval,
            "progress": False,
            "auto_adjust": False,
            "group_by": "column",
            "threads": False,
        }

        if start_date is not None:
            kwargs["start"] = pd.to_datetime(start_date, errors="coerce")
        else:
            kwargs["period"] = period

        df = yf.download(ticker, **kwargs)

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
    if market.period_was_adjusted:
        print(f"Periodo cfg  : {market.configured_period}")
        print(f"Periodo usado: {market.effective_period}")
        print(market.period_adjustment_message)
    else:
        print(f"Periodo      : {market.period}")
    print(f"Intervalo    : {market.interval}")

    variables = CONTEXT_VARIABLES_DEFAULT.copy()

    print(f"Variables contexto: {len(variables)}")
    print("=" * 70)

    datasets = []
    errores = []
    cache = None

    if SQLITE_INCREMENTAL_CACHE:
        cache_db = market.cache_dir / f"{market.market}_cache.sqlite"
        cache = MarketSQLiteCache(cache_db)
        print(f"Cache SQLite      : {cache_db}")
        print(f"Context lookback  : {CONTEXT_RECALC_LOOKBACK_BARS} barras")

    for ticker, nombre in variables.items():
        if cache is None:
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
            continue

        last_cached = cache.get_context_max_date(market.market, market.interval, nombre)
        start_date = None
        if last_cached is not None:
            start_date = last_cached - intervalo_a_lookback_timedelta(market.interval, CONTEXT_RECALC_LOOKBACK_BARS)

        df_new, error = descargar_contexto_ticker(
            ticker=ticker,
            nombre=nombre,
            period=market.period,
            interval=market.interval,
            start_date=start_date,
        )

        if df_new is None:
            df_full = cache.load_context_series(market.market, market.interval, nombre)
            if df_full.empty:
                errores.append(error)
                print(f"ERROR {ticker:12} {error}")
                continue

            print(f"OK    {ticker:12} -> {nombre:12} {len(df_full):6} registros (cache)")
        else:
            upserted = cache.upsert_context_series(market.market, market.interval, nombre, df_new)
            df_full = cache.load_context_series(market.market, market.interval, nombre)
            modo = "full" if last_cached is None else "incremental"
            print(f"OK    {ticker:12} -> {nombre:12} {len(df_full):6} registros ({modo}, upsert={upserted})")

        df_feat = calcular_features_contexto(df_full, nombre)
        datasets.append(df_feat)

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
    contexto_excel = quitar_timezone_para_excel(contexto)

    print()
    print("Guardando CSV de contexto...")
    t_save_csv = datetime.now()
    contexto.to_csv(archivo_csv, index=False, encoding="utf-8-sig")
    print(f"CSV guardado en {datetime.now() - t_save_csv}: {archivo_csv}")

    excel_resumen = "OMITIDO (usar --excel-contexto para habilitar)"
    if exportar_excel_contexto_habilitado():
        print("Guardando Excel de contexto...")
        t_save_xlsx = datetime.now()
        contexto_excel.to_excel(archivo_excel, index=False)
        print(f"Excel guardado en {datetime.now() - t_save_xlsx}: {archivo_excel}")
        excel_resumen = str(archivo_excel)
    else:
        print("Excel de contexto omitido (flag runtime desactivado).")

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
    print(f"Excel generado        : {excel_resumen}")
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