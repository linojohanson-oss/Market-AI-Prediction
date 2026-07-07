import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import pandas as pd
import numpy as np

from config import MarketConfig


TARGET_PRINCIPAL = "Target_5d"


def generar_dataset_ia(nombre_mercado="merval"):
    inicio = datetime.now()
    market = MarketConfig(nombre_mercado)

    print("=" * 70)
    print(f"MARKET DATASET IA - {market.display_name}")
    print("=" * 70)

    if not market.indicators_csv.exists():
        raise FileNotFoundError(
            f"No existe archivo de indicadores: {market.indicators_csv}. "
            f"Primero ejecutá market_indicators.py"
        )

    df = pd.read_csv(market.indicators_csv)
    df["Date"] = pd.to_datetime(df["Date"])

    print(f"Mercado        : {market.market}")
    print(f"Archivo origen : {market.indicators_csv}")
    print(f"Registros base : {len(df):,}")
    print(f"Activos        : {df['Ticker'].nunique()}")

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

    dataset = df.drop(columns=columnas_excluir).copy()

    if TARGET_PRINCIPAL not in dataset.columns:
        raise ValueError(f"No existe columna objetivo {TARGET_PRINCIPAL}")

    columnas_clave = ["Date", "Ticker", TARGET_PRINCIPAL]
    columnas_modelo = [
        c for c in dataset.columns
        if c not in columnas_clave
        and c not in ["Nombre", "Mercado", "Moneda", "Fuente"]
    ]

    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Eliminamos filas sin objetivo y sin suficientes indicadores
    dataset = dataset.dropna(subset=[TARGET_PRINCIPAL])

    # Quitamos filas iniciales donde los indicadores largos todavía no existen
    dataset = (
        dataset
        .groupby("Ticker", group_keys=False)
        .apply(lambda x: x.iloc[200:].copy())
        .reset_index(drop=True)
    )

    # Relleno conservador
    for col in columnas_modelo:
        if col in dataset.columns:
            dataset[col] = pd.to_numeric(dataset[col], errors="coerce")

    dataset[columnas_modelo] = dataset[columnas_modelo].fillna(0)

    dataset.rename(columns={TARGET_PRINCIPAL: "Objetivo_5d"}, inplace=True)

    dataset.sort_values(["Ticker", "Date"], inplace=True)

    dataset.to_csv(market.dataset_csv, index=False, encoding="utf-8-sig")
    dataset.to_excel(market.dataset_excel, index=False)

    objetivo_pct = dataset["Objetivo_5d"].mean()

    print()
    print("=" * 70)
    print(f"RESUMEN DATASET IA - {market.display_name}")
    print("=" * 70)
    print(f"Registros finales     : {len(dataset):,}")
    print(f"Acciones/Activos      : {dataset['Ticker'].nunique()}")
    print(f"Columnas totales      : {len(dataset.columns)}")
    print(f"Columnas modelo       : {len(columnas_modelo)}")
    print(f"Objetivo positivo     : {objetivo_pct:.2%}")
    print(f"Desde                 : {dataset['Date'].min()}")
    print(f"Hasta                 : {dataset['Date'].max()}")
    print(f"CSV generado          : {market.dataset_csv}")
    print(f"Excel generado        : {market.dataset_excel}")
    print(f"Tiempo total          : {datetime.now() - inicio}")

    return dataset


if __name__ == "__main__":
    generar_dataset_ia("crypto")