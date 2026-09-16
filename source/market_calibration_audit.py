from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from config import MarketConfig
from market_training import (
    OBJETIVO,
    crear_split_temporal,
    limpiar_X,
    probabilidades_modelo,
    expected_calibration_error,
)


def auditar_mercado(nombre_mercado="merval"):
    market = MarketConfig(nombre_mercado)
    df = pd.read_csv(market.dataset_csv)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Ticker", OBJETIVO]).copy()
    df.sort_values(["Date", "Ticker"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    train, test, temporal = crear_split_temporal(df)
    features = joblib.load(market.features_pkl)
    X_test = limpiar_X(test, features)
    y_test = test[OBJETIVO].astype(int).to_numpy()

    print("=" * 94)
    print(f"MARKET AI - AUDITORÍA TEMPORAL Y CALIBRACIÓN V2 | {market.display_name}")
    print("=" * 94)
    print(f"Dataset                     : {market.dataset_csv}")
    print(f"Filas                       : {len(df):,}")
    print(f"Train                       : {len(train):,}")
    print(f"Test                        : {len(test):,}")
    print(f"Filas purgadas              : {temporal['filas_purgadas']:,}")
    print(f"Train fechas                : {temporal['train_fecha_min']} -> {temporal['train_fecha_max']}")
    print(f"Test fechas                 : {temporal['test_fecha_min']} -> {temporal['test_fecha_max']}")
    print(f"Fechas presentes en ambos   : {temporal['fechas_solapadas']}")
    print(f"Separación temporal estricta: {temporal['separacion_temporal_estricta']}")
    print(f"Tickers train/test/comunes  : {temporal['tickers_train']}/{temporal['tickers_test']}/{temporal['tickers_comunes']}")

    validacion_dir = market.trained_models_dir / "validacion_temporal"
    if not validacion_dir.exists():
        raise FileNotFoundError(
            f"No existe {validacion_dir}. Ejecutá primero market_training.py; "
            "no se auditan modelos FULL porque ya vieron el período de test."
        )

    metricas_path = market.metrics_dir / "metricas_modelos.csv"
    metricas = pd.read_csv(metricas_path).sort_values("score_modelo", ascending=False)
    resultados = []

    print()
    print("-" * 94)
    print("MODELOS DE VALIDACIÓN TEMPORAL (NO MODELOS FULL)")
    print("-" * 94)

    for nombre in metricas["modelo"].astype(str):
        archivo = validacion_dir / f"{nombre}.pkl"
        if not archivo.exists():
            print(f"OMITIDO {nombre}: no existe {archivo}")
            continue
        try:
            modelo = joblib.load(archivo)
            prob = probabilidades_modelo(modelo, X_test)
            auc = roc_auc_score(y_test, prob)
            fila = {
                "mercado": market.market,
                "modelo": nombre,
                "filas_test": len(y_test),
                "tasa_real": float(np.mean(y_test)),
                "prob_media": float(np.mean(prob)),
                "brecha_media": float(np.mean(prob) - np.mean(y_test)),
                "brier_score": float(brier_score_loss(y_test, prob)),
                "log_loss": float(log_loss(y_test, prob, labels=[0, 1])),
                "roc_auc": float(auc),
                "ece_10": float(expected_calibration_error(y_test, prob)),
            }
            resultados.append(fila)
            print(
                f"{nombre:24} real={fila['tasa_real']:.3f} pred={fila['prob_media']:.3f} "
                f"Brier={fila['brier_score']:.4f} LogLoss={fila['log_loss']:.4f} "
                f"AUC={fila['roc_auc']:.4f} ECE={fila['ece_10']:.4f}"
            )
        except Exception as exc:
            print(f"ERROR {nombre}: {exc}")

    if not resultados:
        raise RuntimeError("No fue posible auditar ningún modelo temporal.")

    salida = pd.DataFrame(resultados).sort_values(
        ["brier_score", "log_loss"], ascending=True
    )
    destino = market.ranking_csv.parent
    destino.mkdir(parents=True, exist_ok=True)
    csv_resumen = destino / "calibracion_modelos_temporal.csv"
    txt_temporal = destino / "auditoria_corte_temporal.txt"
    salida.to_csv(csv_resumen, index=False, encoding="utf-8-sig")
    with open(txt_temporal, "w", encoding="utf-8") as archivo:
        json.dump(temporal, archivo, indent=4, ensure_ascii=False)

    print()
    print("=" * 94)
    print("RESUMEN ORDENADO POR BRIER SCORE (MENOR ES MEJOR)")
    print("=" * 94)
    print(salida[[
        "modelo", "tasa_real", "prob_media", "brecha_media",
        "brier_score", "log_loss", "roc_auc", "ece_10",
    ]].to_string(index=False))
    print()
    print("OK: separación temporal estricta y auditoría sobre modelos de validación.")
    print(f"Resumen calibración : {csv_resumen}")
    print(f"Auditoría temporal  : {txt_temporal}")
    return salida


def main():
    parser = argparse.ArgumentParser(
        description="Audita modelos de validación con corte temporal purgado."
    )
    parser.add_argument("mercado", nargs="?", default="merval")
    args = parser.parse_args()
    auditar_mercado(args.mercado)


if __name__ == "__main__":
    main()
