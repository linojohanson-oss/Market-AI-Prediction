import sys
from pathlib import Path
from datetime import datetime
import warnings

ROOT = Path(__file__).resolve().parent.parent

from source.app_logger import log_print as print

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pandas.api.types import is_datetime64tz_dtype

from config import MarketConfig, get_json_config_value


TARGET_PRINCIPAL = get_json_config_value("pipeline.target_principal", "Target_5d")
TARGET_HORIZONS = get_json_config_value("pipeline.target_horizons", [1, 5, 10, 20, 60])
MIN_ROWS_PER_TICKER = get_json_config_value("pipeline.min_rows_per_ticker", 200)
FILLNA_MODEL_FEATURES = get_json_config_value("pipeline.fillna_model_features", 0)


def resolver_columna(df, posibles):
    for col in posibles:
        if col in df.columns:
            return col
    return None


def normalizar_columnas_dataset(df):
    df = df.copy()

    col_ticker = resolver_columna(df, ["Ticker", "ticker", "Symbol", "symbol"])
    col_fecha = resolver_columna(df, ["Date", "date", "Fecha", "fecha", "Datetime", "datetime"])

    if col_ticker is not None and col_ticker != "Ticker":
        df.rename(columns={col_ticker: "Ticker"}, inplace=True)

    if col_fecha is not None and col_fecha != "Date":
        df.rename(columns={col_fecha: "Date"}, inplace=True)

    return df


def normalizar_fecha_utc_naive(series):
    fechas = pd.to_datetime(series, errors="coerce", utc=True)
    return fechas.dt.tz_localize(None)


def quitar_timezone_para_excel(df):
    df = df.copy()

    for col in df.columns:
        if is_datetime64tz_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    return df


def archivo_contexto(market):
    return market.processed_dir / f"{market.market.upper()}_CONTEXTO.csv"


def cargar_indicadores(market):
    indicadores_files = sorted(
        market.processed_dir.glob(f"{market.market.upper()}_INDICADORES*.csv")
    )

    if not indicadores_files:
        raise FileNotFoundError(
            f"No existe archivo de indicadores en {market.processed_dir}. "
            f"Primero ejecutá market_indicators.py"
        )

    indicadores_path = indicadores_files[-1]

    if not indicadores_path.exists():
        raise FileNotFoundError(
            f"No existe archivo de indicadores: {indicadores_path}. "
            f"Primero ejecutá market_indicators.py"
        )

    df = pd.read_csv(indicadores_path)
    df = normalizar_columnas_dataset(df)

    if "Date" not in df.columns:
        raise ValueError(f"No encontré columna de fecha en {indicadores_path}")

    if "Ticker" not in df.columns:
        raise ValueError(f"No encontré columna de ticker en {indicadores_path}")

    df["Date"] = normalizar_fecha_utc_naive(df["Date"])
    df = df.dropna(subset=["Date"]).copy()

    return df


def cargar_contexto(market):
    contexto_csv = archivo_contexto(market)

    if not contexto_csv.exists():
        print(f"ADVERTENCIA: no existe contexto macro: {contexto_csv}")
        print("Se generará dataset solo con indicadores técnicos.")
        return None
    print(f"Archivo Contexto { contexto_csv}")
    contexto = pd.read_csv(contexto_csv)
    contexto["Date"] = normalizar_fecha_utc_naive(contexto["Date"])
    contexto = contexto.dropna(subset=["Date"]).copy()

    contexto.sort_values("Date", inplace=True)
    contexto.drop_duplicates(subset=["Date"], keep="last", inplace=True)

    return contexto


def integrar_contexto(df, contexto):
    if contexto is None:
        return df, 0

    columnas_contexto = [c for c in contexto.columns if c != "Date"]

    df = df.sort_values("Date").copy()
    contexto = contexto.sort_values("Date").copy()

    integrado = pd.merge_asof(
        df,
        contexto,
        on="Date",
        direction="backward",
    )

    integrado[columnas_contexto] = integrado[columnas_contexto].ffill().bfill()

    return integrado, len(columnas_contexto)


def limpiar_targets_futuros(df):
    columnas_excluir = []

    for horizon in TARGET_HORIZONS:
        if horizon == 5:
            continue
        columnas_excluir.extend([
            f"Future_Close_{horizon}d",
            f"Future_Return_{horizon}d",
            f"Target_{horizon}d",
        ])

    columnas_excluir = [c for c in columnas_excluir if c in df.columns]

    return df.drop(columns=columnas_excluir).copy()


def generar_dataset_ia(nombre_mercado="merval"):
    inicio = datetime.now()
    market = MarketConfig(nombre_mercado)

    print("=" * 70)
    print(f"MARKET DATASET IA + CONTEXTO - {market.display_name}")
    print("=" * 70)

    df = cargar_indicadores(market)

    print(f"Mercado          : {market.market}")
    print(f"Indicadores      : {market.indicators_csv}")
    print(f"Registros base   : {len(df):,}")
    print(f"Activos          : {df['Ticker'].nunique()}")

    contexto = cargar_contexto(market)

    if contexto is not None:
        print(f"Contexto macro   : {archivo_contexto(market)}")
        print(f"Registros contexto: {len(contexto):,}")
        print(f"Columnas contexto : {len(contexto.columns) - 1}")
    else:
        print("Contexto macro   : NO DISPONIBLE")

    df, columnas_contexto = integrar_contexto(df, contexto)

    dataset = limpiar_targets_futuros(df)

    if TARGET_PRINCIPAL not in dataset.columns:
        raise ValueError(f"No existe columna objetivo {TARGET_PRINCIPAL}")

    dataset.rename(columns={TARGET_PRINCIPAL: "Objetivo_5d"}, inplace=True)

    columnas_excluir_modelo = [
        "Date",
        "Ticker",
        "Nombre",
        "Mercado",
        "Moneda",
        "Fuente",
        "Objetivo_5d",
    ]

    columnas_modelo = [
        c for c in dataset.columns
        if c not in columnas_excluir_modelo
    ]

    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataset = dataset.dropna(subset=["Objetivo_5d"])

    frames = []
    for ticker, grupo in dataset.groupby("Ticker", sort=False):
        if len(grupo) > MIN_ROWS_PER_TICKER:
            frames.append(grupo.iloc[MIN_ROWS_PER_TICKER:].copy())
        else:
            frames.append(grupo.copy())

    if frames:
        dataset = pd.concat(frames, ignore_index=True)
    else:
        dataset = dataset.copy()

    for col in columnas_modelo:
        if col in dataset.columns:
            dataset[col] = pd.to_numeric(dataset[col], errors="coerce")

    dataset[columnas_modelo] = dataset[columnas_modelo].fillna(FILLNA_MODEL_FEATURES)

    dataset.sort_values(["Ticker", "Date"], inplace=True)
    dataset_excel = quitar_timezone_para_excel(dataset)

    print()
    print("Guardando CSV de dataset IA...")
    t_save_csv = datetime.now()
    dataset.to_csv(market.dataset_csv, index=False, encoding="utf-8-sig")
    print(f"CSV guardado en {datetime.now() - t_save_csv}: {market.dataset_csv}")

    print("Guardando Excel de dataset IA...")
    t_save_xlsx = datetime.now()
    dataset_excel.to_excel(market.dataset_excel, index=False)
    print(f"Excel guardado en {datetime.now() - t_save_xlsx}: {market.dataset_excel}")

    objetivo_pct = dataset["Objetivo_5d"].mean()

    print()
    print("=" * 70)
    print(f"RESUMEN DATASET IA - {market.display_name}")
    print("=" * 70)
    print(f"Registros finales       : {len(dataset):,}")
    print(f"Acciones/Activos        : {dataset['Ticker'].nunique()}")
    print(f"Columnas totales        : {len(dataset.columns)}")
    print(f"Columnas modelo         : {len(columnas_modelo)}")
    print(f"Columnas contexto       : {columnas_contexto}")
    print(f"Objetivo positivo       : {objetivo_pct:.2%}")
    print(f"Desde                   : {dataset['Date'].min()}")
    print(f"Hasta                   : {dataset['Date'].max()}")
    print(f"CSV generado            : {market.dataset_csv}")
    print(f"Excel generado          : {market.dataset_excel}")
    print(f"Tiempo total            : {datetime.now() - inicio}")

    return dataset


if __name__ == "__main__":
    generar_dataset_ia("merval")