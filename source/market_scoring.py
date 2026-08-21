import sys
from pathlib import Path
from datetime import datetime
import warnings

ROOT = Path(__file__).resolve().parent.parent

from source.app_logger import log_print as print

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from source.market_model_manager import MarketModelManager


TOP_MODELOS = 3


def preparar_ultimos_datos(manager):
    df = manager.cargar_dataset()
    features = manager.cargar_features()

    ultima_fecha = df["Date"].max()
    ultimos = df[df["Date"] == ultima_fecha].copy()

    X = ultimos[features].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    return df, ultimos, X, ultima_fecha


def probabilidad_modelo(modelo, X):
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]

    pred = modelo.predict(X)
    return pred.astype(float)


def convertir_a_score_relativo(matriz_probs):
    scores = []

    for i in range(matriz_probs.shape[1]):
        serie = pd.Series(matriz_probs[:, i])
        score = serie.rank(pct=True).values * 100
        scores.append(score)

    return np.vstack(scores).T


def calcular_consenso(ultimos, modelos, X):
    modelos = sorted(
        modelos,
        key=lambda x: x.get("peso", 0),
        reverse=True
    )[:TOP_MODELOS]

    print()
    print("=" * 70)
    print(f"MODELOS USADOS EN SCORING: TOP {TOP_MODELOS}")
    print("=" * 70)

    matriz_probs = []
    pesos = []
    nombres = []

    for item in modelos:
        nombre = item["modelo"]
        peso = item["peso"]
        modelo = item["objeto"]

        try:
            prob = probabilidad_modelo(modelo, X)

            matriz_probs.append(prob)
            pesos.append(peso)
            nombres.append(nombre)

            ultimos[f"Prob_{nombre}"] = prob

            print(f"{nombre:25} Peso: {peso:.4f}")

        except Exception as e:
            print(f"Error calculando probabilidad para {nombre}: {e}")

    if not matriz_probs:
        raise RuntimeError("No se pudieron calcular probabilidades.")

    matriz_probs = np.vstack(matriz_probs).T

    pesos = np.array(pesos)
    pesos = pesos / pesos.sum()

    consensus_probability = np.average(
        matriz_probs,
        axis=1,
        weights=pesos
    )

    matriz_scores_relativos = convertir_a_score_relativo(matriz_probs)

    score_relativo = np.average(
        matriz_scores_relativos,
        axis=1,
        weights=pesos
    )

    score_absoluto = consensus_probability * 100

    # Combina ranking relativo con probabilidad absoluta.
    # Esto evita señales demasiado agresivas cuando todo el mercado está débil.
    consensus_score = (
        score_relativo * 0.70
        + score_absoluto * 0.30
    )

    dispersion = np.std(matriz_scores_relativos, axis=1)
    uncertainty_score = dispersion
    agreement_score = 100 - uncertainty_score

    percentiles = np.percentile(matriz_scores_relativos, [33, 66], axis=0)

    umbral_venta = percentiles[0]
    umbral_compra = percentiles[1]

    votos_compra = (matriz_scores_relativos >= umbral_compra).sum(axis=1)
    votos_venta = (matriz_scores_relativos <= umbral_venta).sum(axis=1)
    votos_neutral = matriz_scores_relativos.shape[1] - votos_compra - votos_venta

    ultimos["Consensus_Probability"] = consensus_probability
    ultimos["Score_Relativo"] = score_relativo
    ultimos["Score_Absoluto"] = score_absoluto
    ultimos["Consensus_Score"] = consensus_score
    ultimos["Model_Dispersion"] = dispersion
    ultimos["Agreement_Score"] = agreement_score
    ultimos["Uncertainty_Score"] = uncertainty_score
    ultimos["Buy_Votes"] = votos_compra
    ultimos["Neutral_Votes"] = votos_neutral
    ultimos["Sell_Votes"] = votos_venta
    ultimos["Models_Used"] = matriz_scores_relativos.shape[1]

    return ultimos


def calcular_score_tecnico(df):
    df = df.copy()

    def col(nombre, default=0):
        return df[nombre] if nombre in df.columns else default

    rsi = col("RSI_14", 50)
    macd = col("MACD", 0)
    macd_signal = col("MACD_Signal", 0)
    volume_ratio = col("Volume_Ratio", 1)
    ret_20 = col("Return_20d", 0)

    ema20 = col("EMA_20", df["Close"])
    ema50 = col("EMA_50", df["Close"])
    ema200 = col("EMA_200", df["Close"])

    df["Score_RSI"] = np.where(
        rsi.between(45, 70),
        100,
        np.where(rsi < 45, 60, 40),
    )

    df["Score_Tendencia"] = np.where(
        (df["Close"] > ema20) & (ema20 > ema50) & (ema50 > ema200),
        100,
        np.where(df["Close"] > ema50, 70, 40),
    )

    df["Score_MACD"] = np.where(macd > macd_signal, 100, 40)

    df["Score_Volumen"] = np.where(
        volume_ratio > 1.2,
        100,
        np.where(volume_ratio > 0.8, 70, 40),
    )

    df["Score_Momentum"] = np.where(ret_20 > 0, 100, 40)

    df["Score_Tecnico"] = (
        df["Score_RSI"] * 0.20
        + df["Score_Tendencia"] * 0.30
        + df["Score_MACD"] * 0.20
        + df["Score_Volumen"] * 0.15
        + df["Score_Momentum"] * 0.15
    )

    return df


def clasificar_senal(df):
    df = df.copy()

    df["Score_Final"] = (
        df["Consensus_Score"] * 0.50
        + df["Score_Tecnico"] * 0.35
        + df["Agreement_Score"] * 0.15
    )

    df["Fortaleza_Senal"] = np.select(
        [
            df["Score_Final"] >= 80,
            df["Score_Final"] >= 70,
            df["Score_Final"] >= 60,
            df["Score_Final"] >= 50,
        ],
        [
            "MUY FUERTE",
            "FUERTE",
            "MEDIA",
            "DÉBIL",
        ],
        default="MUY DÉBIL",
    )

    df["Señal"] = np.select(
        [
            (df["Score_Final"] >= 75)
            & (df["Buy_Votes"] >= df["Models_Used"] * 0.60)
            & (df["Consensus_Probability"] >= 0.10),

            (df["Score_Final"] >= 65)
            & (df["Buy_Votes"] >= df["Models_Used"] * 0.50)
            & (df["Consensus_Probability"] >= 0.07),

            df["Score_Final"] >= 55,

            df["Score_Final"] >= 45,
        ],
        [
            "COMPRA FUERTE",
            "COMPRA",
            "VIGILAR",
            "NEUTRAL",
        ],
        default="EVITAR",
    )

    return df


def ordenar_columnas(df):
    columnas_base = [
        "Date",
        "Ticker",
        "Nombre",
        "Close",
        "Consensus_Probability",
        "Consensus_Score",
        "Score_Relativo",
        "Score_Absoluto",
        "Agreement_Score",
        "Uncertainty_Score",
        "Model_Dispersion",
        "Buy_Votes",
        "Neutral_Votes",
        "Sell_Votes",
        "Models_Used",
        "Score_Tecnico",
        "Score_Final",
        "Fortaleza_Senal",
        "Señal",
    ]

    columnas_modelos = [c for c in df.columns if c.startswith("Prob_")]

    columnas_extra = [
        "RSI_14",
        "Return_5d",
        "Return_20d",
        "Volume_Ratio",
        "Score_RSI",
        "Score_Tendencia",
        "Score_MACD",
        "Score_Volumen",
        "Score_Momentum",
    ]

    columnas = columnas_base + columnas_modelos + columnas_extra
    columnas = [c for c in columnas if c in df.columns]

    resto = [c for c in df.columns if c not in columnas]

    return df[columnas + resto]


def generar_market_scoring(nombre_mercado="merval"):
    inicio = datetime.now()

    manager = MarketModelManager(nombre_mercado)
    market = manager.market

    print("=" * 70)
    print(f"MARKET SCORING TOP {TOP_MODELOS} MODELOS - {market.display_name}")
    print("=" * 70)

    manager.guardar_modelo_activo()
    manager.guardar_ensemble()
    manager.guardar_ranking_modelos()

    df, ultimos, X, ultima_fecha = preparar_ultimos_datos(manager)
    modelos = manager.cargar_modelos_ensemble()

    print(f"Mercado        : {market.market}")
    print(f"Fecha scoring  : {ultima_fecha}")
    print(f"Activos        : {len(ultimos)}")
    print(f"Modelos base   : {len(modelos)}")

    ultimos = calcular_consenso(ultimos, modelos, X)
    ultimos = calcular_score_tecnico(ultimos)
    ultimos = clasificar_senal(ultimos)

    ultimos.sort_values("Score_Final", ascending=False, inplace=True)
    ultimos.reset_index(drop=True, inplace=True)

    ranking = ordenar_columnas(ultimos)

    ranking.to_csv(market.ranking_csv, index=False, encoding="utf-8-sig")
    ranking.to_excel(market.ranking_excel, index=False)

    print()
    print("=" * 70)
    print(f"TOP OPORTUNIDADES - {market.display_name}")
    print("=" * 70)

    print(
        ranking[
            [
                "Ticker",
                "Close",
                "Consensus_Probability",
                "Consensus_Score",
                "Score_Relativo",
                "Score_Absoluto",
                "Agreement_Score",
                "Buy_Votes",
                "Models_Used",
                "Score_Tecnico",
                "Score_Final",
                "Señal",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print()
    print(f"CSV generado   : {market.ranking_csv}")
    print(f"Excel generado : {market.ranking_excel}")
    print(f"Tiempo total   : {datetime.now() - inicio}")

    return ranking


if __name__ == "__main__":
    generar_market_scoring("crypto")