import sys
from pathlib import Path
from datetime import datetime
import warnings

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from config import MarketConfig


TARGET_PRINCIPAL = "Target_5d"


def archivo_contexto(market):
    return market.processed_dir / f"{market.market.upper()}_CONTEXTO.csv"


def cargar_indicadores(market):
    if not market.indicators_csv.exists():
        raise FileNotFoundError(
            f"No existe archivo de indicadores: {market.indicators_csv}. "
            f"Primero ejecutá market_indicators.py"
        )

    df = pd.read_csv(market.indicators_csv)
    df["Date"] = pd.to_datetime(df["Date"])

    return df


def cargar_contexto(market):
    contexto_csv = archivo_contexto(market)

    if not contexto_csv.exists():
        print(f"ADVERTENCIA: no existe contexto macro: {contexto_csv}")
        print("Se generará dataset solo con indicadores técnicos.")
        return None

    contexto = pd.read_csv(contexto_csv)
    contexto["Date"] = pd.to_datetime(contexto["Date"])

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
    columnas_excluir = [
        "Future_Close_1d",
        "Future_Return_1d",
        "Target_1d",
        "Future_Close_5d",
        "Future_Return_5d",
        "Future_Close_10d",
        "Future_Return_10d",
        "Target_10d",
        "Future_Close_20d",
        "Future_Return_20d",
        "Target_20d",
        "Future_Close_60d",
        "Future_Return_60d",
        "Target_60d",
    ]

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

    dataset = (
        dataset
        .groupby("Ticker", group_keys=False)
        .apply(lambda x: x.iloc[200:].copy())
        .reset_index(drop=True)
    )

    for col in columnas_modelo:
        if col in dataset.columns:
            dataset[col] = pd.to_numeric(dataset[col], errors="coerce")

    dataset[columnas_modelo] = dataset[columnas_modelo].fillna(0)

    dataset.sort_values(["Ticker", "Date"], inplace=True)

    dataset.to_csv(market.dataset_csv, index=False, encoding="utf-8-sig")
    dataset.to_excel(market.dataset_excel, index=False)

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