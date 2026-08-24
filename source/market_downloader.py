import sys
import os
from pathlib import Path
from datetime import datetime
import sqlite3
import json

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


class RawMarketSQLiteCache:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _ensure_schema(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS raw_rows (
                    market TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (market, interval, ticker, date)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_raw_lookup
                ON raw_rows (market, interval, ticker, date)
                """
            )

    @staticmethod
    def _to_datetime_utc(values):
        try:
            return pd.to_datetime(values, errors="coerce", utc=True, format="mixed")
        except TypeError:
            return pd.to_datetime(values, errors="coerce", utc=True)

    @staticmethod
    def _to_iso(v):
        ts = RawMarketSQLiteCache._to_datetime_utc(v)
        return ts.isoformat() if not pd.isna(ts) else "NaT"

    def get_raw_max_date(self, market: str, interval: str, ticker: str):
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT MAX(date) FROM raw_rows
                WHERE market = ? AND interval = ? AND ticker = ?
                """,
                (market, interval, ticker),
            ).fetchone()

        if not row or not row[0]:
            return None
        return self._to_datetime_utc(row[0])

    def upsert_raw_rows(self, market: str, interval: str, ticker: str, df: pd.DataFrame):
        if df is None or df.empty:
            return 0

        working = df.copy()
        if "Date" not in working.columns:
            return 0

        working["Date"] = self._to_datetime_utc(working["Date"])
        working.dropna(subset=["Date"], inplace=True)

        records = []
        updated_at = datetime.now().isoformat()

        for _, row in working.iterrows():
            d = row.to_dict()
            date_iso = self._to_iso(d.get("Date"))
            if not date_iso or date_iso == "NaT":
                continue

            payload = dict(d)
            payload.pop("Date", None)
            payload.pop("Ticker", None)

            records.append(
                (
                    market,
                    interval,
                    ticker,
                    date_iso,
                    json.dumps(payload, ensure_ascii=False, default=str),
                    updated_at,
                )
            )

        if not records:
            return 0

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO raw_rows (market, interval, ticker, date, payload, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(market, interval, ticker, date)
                DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at
                """,
                records,
            )

        return len(records)

    def load_raw_ticker(self, market: str, interval: str, ticker: str):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT date, payload
                FROM raw_rows
                WHERE market = ? AND interval = ? AND ticker = ?
                ORDER BY date
                """,
                (market, interval, ticker),
            ).fetchall()

        if not rows:
            return pd.DataFrame()

        out = []
        for date_iso, payload_json in rows:
            payload = json.loads(payload_json)
            payload["Date"] = self._to_datetime_utc(date_iso)
            payload["Ticker"] = ticker
            out.append(payload)

        df = pd.DataFrame(out)
        if "Date" in df.columns:
            df["Date"] = self._to_datetime_utc(df["Date"])
        return df


def _parse_bool_env(value, default=False):
    if value is None:
        return default

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def exportar_excel_descarga_habilitado():
    # Runtime opt-in: disabled by default for speed and process stability.
    global_default = _parse_bool_env(os.getenv("MARKET_EXPORT_EXCEL"), default=False)
    return _parse_bool_env(os.getenv("MARKET_DOWNLOADER_EXPORT_EXCEL"), default=global_default)


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


def descargar_ticker(market, ticker, nombre, raw_cache=None):
    inicio = datetime.now()

    try:
        last_cached = None
        if raw_cache is not None:
            last_cached = raw_cache.get_raw_max_date(market.market, market.interval, ticker)

        yf_kwargs = {
            "interval": market.interval,
            "progress": False,
            "auto_adjust": False,
            "group_by": "column",
            "threads": False,
        }
        skip_download = False

        if last_cached is None:
            yf_kwargs["period"] = market.period
        else:
            start_ts = pd.to_datetime(last_cached, errors="coerce", utc=True)
            if pd.isna(start_ts):
                yf_kwargs["period"] = market.period
            else:
                # Fetch only data after the last cached candle.
                start_ts = start_ts.tz_localize(None) if start_ts.tzinfo is not None else start_ts
                start_candidate = (start_ts + pd.Timedelta(days=1)).date()
                today_utc = pd.Timestamp.now(tz="UTC").date()

                if start_candidate > today_utc:
                    skip_download = True
                else:
                    yf_kwargs["start"] = start_candidate.strftime("%Y-%m-%d")

        if skip_download:
            df = pd.DataFrame()
        else:
            df = yf.download(ticker, **yf_kwargs)

        if (df is None or df.empty) and raw_cache is None:
            return {
                "ticker": ticker,
                "ok": False,
                "error": "Sin datos",
                "registros": 0,
                "tiempo": datetime.now() - inicio,
            }

        if df is not None and not df.empty:
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

        if raw_cache is not None:
            if df is not None and not df.empty:
                raw_cache.upsert_raw_rows(market.market, market.interval, ticker, df)
            df = raw_cache.load_raw_ticker(market.market, market.interval, ticker)
            if df.empty:
                return {
                    "ticker": ticker,
                    "ok": False,
                    "error": "Sin datos (ni descarga nueva ni caché histórica)",
                    "registros": 0,
                    "tiempo": datetime.now() - inicio,
                }

        nombre_archivo = limpiar_nombre_archivo(ticker)

        archivo_csv = market.raw_dir / f"{nombre_archivo}.csv"
        archivo_excel = market.raw_dir / f"{nombre_archivo}.xlsx"

        df.to_csv(archivo_csv, index=False, encoding="utf-8-sig")
        if exportar_excel_descarga_habilitado():
            df_excel = quitar_timezone_para_excel(df)
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
    raw_cache_db = market.cache_dir / f"{market.market}_raw_cache.sqlite"
    raw_cache = RawMarketSQLiteCache(raw_cache_db)
    print(f"Raw cache DB  : {raw_cache_db}")
    print("=" * 70)

    resultados = []
    dataframes = []

    symbols = list(market.symbols.items())

    total = len(symbols)

    for idx, (ticker, nombre) in enumerate(symbols, start=1):
        resultado = descargar_ticker(market, ticker, nombre, raw_cache=raw_cache)
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
    excel_resumen = "OMITIDO (usar --excel-download para habilitar)"
    if exportar_excel_descarga_habilitado():
        base_excel.to_excel(market.full_raw_excel, index=False)
        excel_resumen = str(market.full_raw_excel)

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
    print(f"Excel generado      : {excel_resumen}")
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