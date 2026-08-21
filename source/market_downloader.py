import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent

from source.app_logger import log_print as print

import pandas as pd
import yfinance as yf
from pandas.api.types import is_datetime64tz_dtype

from config import MarketConfig


# IMPORTANTE:
# yfinance puede mezclar datos si se usa en paralelo con algunos tickers.
# Por seguridad dejamos descarga secuencial y threads=False.
MAX_WORKERS = 1


def limpiar_nombre_archivo(ticker):
    return (
        ticker
        .replace(".", "_")
        .replace("=", "_")
        .replace("^", "")
        .replace("/", "_")
    )


def normalizar_columnas(df):
    if isinstance(df.columns, pd.MultiIndex):
        nuevas = []

        for col in df.columns:
            partes = [str(x).strip() for x in col if str(x).strip()]
            nuevas.append(partes[0] if partes else "columna")

        df.columns = nuevas

    df.columns = [str(c).strip() for c in df.columns]

    # Evita columnas duplicadas después de aplanar MultiIndex
    df = df.loc[:, ~df.columns.duplicated()]

    return df


def validar_dataframe(df, ticker):
    if df is None or df.empty:
        return False, f"{ticker}: dataframe vacío"

    columnas_minimas = ["Date", "Open", "High", "Low", "Close"]
    faltantes = [c for c in columnas_minimas if c not in df.columns]

    if faltantes:
        return False, f"{ticker}: faltan columnas {faltantes}"

    fechas = pd.to_datetime(df["Date"], errors="coerce")
    close = pd.to_numeric(df["Close"], errors="coerce")

    if fechas.isna().all():
        return False, f"{ticker}: fechas inválidas"

    if close.isna().all():
        return False, f"{ticker}: precios Close inválidos"

    # Validación básica OHLC
    for col in ["Open", "High", "Low", "Close"]:
        serie = pd.to_numeric(df[col], errors="coerce")
        if serie.isna().all():
            return False, f"{ticker}: columna {col} inválida"

    return True, "OK"


def quitar_timezone_para_excel(df):
    df = df.copy()

    for col in df.columns:
        if is_datetime64tz_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    return df


def descargar_ticker(market, ticker, nombre):
    inicio = datetime.now()

    try:
        df = yf.download(
            ticker,
            period=market.period,
            interval=market.interval,
            progress=False,
            auto_adjust=False,
            group_by="column",
            threads=False,
        )

        if df is None or df.empty:
            return {
                "ticker": ticker,
                "ok": False,
                "error": "Sin datos",
                "registros": 0,
                "tiempo": datetime.now() - inicio,
            }

        df = normalizar_columnas(df)
        df.reset_index(inplace=True)

        # A veces yfinance devuelve Datetime en lugar de Date
        if "Datetime" in df.columns and "Date" not in df.columns:
            df.rename(columns={"Datetime": "Date"}, inplace=True)

        df["Ticker"] = ticker
        df["Nombre"] = nombre
        df["Mercado"] = market.market
        df["Moneda"] = market.currency
        df["Fuente"] = market.source

        columnas_base = [
            "Date",
            "Ticker",
            "Nombre",
            "Mercado",
            "Moneda",
            "Fuente",
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
        ]

        columnas = [c for c in columnas_base if c in df.columns]
        df = df[columnas].copy()

        valido, mensaje = validar_dataframe(df, ticker)

        if not valido:
            return {
                "ticker": ticker,
                "ok": False,
                "error": mensaje,
                "registros": 0,
                "tiempo": datetime.now() - inicio,
            }

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df.dropna(subset=["Date", "Close"], inplace=True)
        df.sort_values("Date", inplace=True)
        df.drop_duplicates(subset=["Date"], keep="last", inplace=True)

        nombre_archivo = limpiar_nombre_archivo(ticker)

        archivo_csv = market.raw_dir / f"{nombre_archivo}.csv"
        archivo_excel = market.raw_dir / f"{nombre_archivo}.xlsx"
        df_excel = quitar_timezone_para_excel(df)

        df.to_csv(archivo_csv, index=False, encoding="utf-8-sig")
        df_excel.to_excel(archivo_excel, index=False)

        return {
            "ticker": ticker,
            "ok": True,
            "error": "",
            "registros": len(df),
            "data": df,
            "archivo_csv": archivo_csv,
            "archivo_excel": archivo_excel,
            "tiempo": datetime.now() - inicio,
        }

    except Exception as e:
        return {
            "ticker": ticker,
            "ok": False,
            "error": str(e),
            "registros": 0,
            "tiempo": datetime.now() - inicio,
        }


def descargar_mercado(nombre_mercado="merval"):
    inicio_total = datetime.now()
    market = MarketConfig(nombre_mercado)

    print("=" * 70)
    print(f"MARKET DOWNLOADER - {market.display_name}")
    print("=" * 70)
    print(f"Mercado       : {market.market}")
    print(f"Descripción   : {market.description}")
    print(f"Activos       : {len(market.tickers)}")
    if market.period_was_adjusted:
        print(f"Periodo cfg   : {market.configured_period}")
        print(f"Periodo usado : {market.effective_period}")
        print(market.period_adjustment_message)
    else:
        print(f"Periodo       : {market.period}")
    print(f"Intervalo     : {market.interval}")
    print(f"Fuente        : {market.source}")
    print(f"Destino       : {market.raw_dir}")
    print("=" * 70)

    resultados = []
    dataframes = []

    symbols = list(market.symbols.items())

    total = len(symbols)

    for idx, (ticker, nombre) in enumerate(symbols, start=1):
        resultado = descargar_ticker(market, ticker, nombre)
        resultados.append(resultado)

        if resultado["ok"]:
            dataframes.append(resultado["data"])

            print(
                f"[{idx:02d}/{total:02d}] OK    "
                f"{resultado['ticker']:12} "
                f"{resultado['registros']:6} registros "
                f"{resultado['tiempo']}"
            )
        else:
            print(
                f"[{idx:02d}/{total:02d}] ERROR "
                f"{resultado['ticker']:12} "
                f"{resultado['error']}"
            )

    if not dataframes:
        print()
        print("No se descargaron datos válidos.")
        return None

    base = pd.concat(dataframes, ignore_index=True)
    base["Date"] = pd.to_datetime(base["Date"], errors="coerce")
    base.dropna(subset=["Date", "Close"], inplace=True)
    base.sort_values(["Ticker", "Date"], inplace=True)
    base.drop_duplicates(subset=["Ticker", "Date"], keep="last", inplace=True)

    base_excel = quitar_timezone_para_excel(base)

    base.to_csv(market.full_raw_csv, index=False, encoding="utf-8-sig")
    base_excel.to_excel(market.full_raw_excel, index=False)

    errores = [r for r in resultados if not r["ok"]]
    exitos = [r for r in resultados if r["ok"]]

    print()
    print("=" * 70)
    print(f"RESUMEN DESCARGA - {market.display_name}")
    print("=" * 70)
    print(f"Activos solicitados : {len(symbols)}")
    print(f"Descargados OK      : {len(exitos)}")
    print(f"Errores             : {len(errores)}")
    print(f"Registros totales   : {len(base):,}")
    print(f"Desde               : {base['Date'].min()}")
    print(f"Hasta               : {base['Date'].max()}")
    print(f"CSV generado        : {market.full_raw_csv}")
    print(f"Excel generado      : {market.full_raw_excel}")
    print(f"Tiempo total        : {datetime.now() - inicio_total}")

    if errores:
        print()
        print("Errores detectados")
        print("-" * 70)

        for e in errores:
            print(f"{e['ticker']}: {e['error']}")

    return base


if __name__ == "__main__":
    descargar_mercado("merval")