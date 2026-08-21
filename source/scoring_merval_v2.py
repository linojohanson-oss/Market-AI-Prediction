import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from source.app_logger import log_print as print

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
    BASE_DIR, "datos", "merval", "procesados", "MERVAL_DATASET_IA.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "modelos", "merval", "BEST_MODEL_MERVAL.pkl"
)

FEATURE_COLUMNS_PATH = os.path.join(
    BASE_DIR, "modelos", "merval", "FEATURE_COLUMNS_MERVAL.pkl"
)

IMPUTE_VALUES_PATH = os.path.join(
    BASE_DIR, "modelos", "merval", "IMPUTE_VALUES_MERVAL.pkl"
)

REPORTES_DIR = os.path.join(BASE_DIR, "reportes", "merval")
os.makedirs(REPORTES_DIR, exist_ok=True)


def clasificar_confianza(prob):
    if prob >= 0.85:
        return "MUY ALTA", "★★★★★"
    elif prob >= 0.75:
        return "ALTA", "★★★★☆"
    elif prob >= 0.65:
        return "MEDIA", "★★★☆☆"
    elif prob >= 0.55:
        return "BAJA", "★★☆☆☆"
    else:
        return "MUY BAJA", "★☆☆☆☆"


def generar_senal(prob, score_final):
    if prob >= 0.80 and score_final >= 80:
        return "COMPRA FUERTE"
    elif prob >= 0.70 and score_final >= 70:
        return "COMPRA"
    elif prob >= 0.60:
        return "VIGILAR"
    elif prob >= 0.50:
        return "NEUTRAL"
    else:
        return "EVITAR"


def calcular_score_tecnico(row):
    score = 50

    if row.get("Close_vs_SMA20", 0) > 0:
        score += 10
    if row.get("Close_vs_SMA50", 0) > 0:
        score += 10
    if row.get("Close_vs_SMA200", 0) > 0:
        score += 10
    if row.get("Trend_SMA20_50", 0) > 0:
        score += 10
    if row.get("Trend_SMA50_200", 0) > 0:
        score += 10

    rsi = row.get("RSI_14", 50)

    if 45 <= rsi <= 65:
        score += 10
    elif rsi > 75:
        score -= 10

    return max(0, min(100, score))


def preparar_features_scoring(df_ult):
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    impute_values = joblib.load(IMPUTE_VALUES_PATH)

    X = df_ult.copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.select_dtypes(include=[np.number])

    X = X.reindex(columns=feature_columns)
    X = X.fillna(impute_values)
    X = X.fillna(0)

    print(f"Features esperadas por modelo: {len(feature_columns)}")
    print(f"Features usadas en scoring : {X.shape[1]}")

    return X


def generar_scoring_v2():
    print("=" * 80)
    print("SCORING ENGINE V2 - MERVAL")
    print("=" * 80)

    for path, nombre in [
        (DATASET_PATH, "dataset"),
        (MODEL_PATH, "modelo"),
        (FEATURE_COLUMNS_PATH, "features"),
        (IMPUTE_VALUES_PATH, "imputación"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"No existe {nombre}: {path}")

    df = pd.read_csv(DATASET_PATH)
    modelo = joblib.load(MODEL_PATH)

    df["Date"] = pd.to_datetime(df["Date"])
    ultima_fecha = df["Date"].max()

    df_ult = df[df["Date"] == ultima_fecha].copy()

    print(f"Dataset cargado : {len(df):,} registros")
    print(f"Última fecha    : {ultima_fecha.date()}")
    print(f"Acciones        : {df_ult['Ticker'].nunique()}")

    X = preparar_features_scoring(df_ult)

    if hasattr(modelo, "predict_proba"):
        probs = modelo.predict_proba(X)[:, 1]
    else:
        probs = modelo.predict(X)

    df_ult["Prob_IA"] = probs
    df_ult["Score_IA"] = df_ult["Prob_IA"] * 100
    df_ult["Score_Tecnico"] = df_ult.apply(calcular_score_tecnico, axis=1)

    df_ult["Score_Final"] = (
        df_ult["Score_IA"] * 0.65 +
        df_ult["Score_Tecnico"] * 0.35
    )

    confianza = df_ult["Prob_IA"].apply(clasificar_confianza)
    df_ult["Confianza"] = confianza.apply(lambda x: x[0])
    df_ult["Rating"] = confianza.apply(lambda x: x[1])

    df_ult["Señal"] = df_ult.apply(
        lambda row: generar_senal(row["Prob_IA"], row["Score_Final"]),
        axis=1
    )

    ranking = df_ult[
        [
            "Date",
            "Ticker",
            "Nombre",
            "Close",
            "Prob_IA",
            "Score_IA",
            "Score_Tecnico",
            "Score_Final",
            "Confianza",
            "Rating",
            "Señal",
        ]
    ].sort_values("Score_Final", ascending=False)

    csv_path = os.path.join(REPORTES_DIR, "ranking_diario_merval_v2.csv")
    xlsx_path = os.path.join(REPORTES_DIR, "ranking_diario_merval_v2.xlsx")

    ranking.to_csv(csv_path, index=False)
    ranking.to_excel(xlsx_path, index=False)

    print()
    print("=" * 80)
    print("TOP OPORTUNIDADES - SCORING V2")
    print("=" * 80)
    print(ranking.head(10).to_string(index=False))

    print()
    print(f"CSV generado   : {csv_path}")
    print(f"Excel generado : {xlsx_path}")
    print(f"Tiempo         : {datetime.now()}")


if __name__ == "__main__":
    generar_scoring_v2()