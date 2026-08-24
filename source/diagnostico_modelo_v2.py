import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from source.app_logger import log_print as print


def _parse_bool_env(value, default=False):
    if value is None:
        return default

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def exportar_excel_diagnostico_habilitado():
    global_default = _parse_bool_env(os.getenv("MARKET_EXPORT_EXCEL"), default=False)
    return _parse_bool_env(os.getenv("MARKET_DIAGNOSTICO_EXPORT_EXCEL"), default=global_default)

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


def preparar_x(df):
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    impute_values = joblib.load(IMPUTE_VALUES_PATH)

    X = df.copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.select_dtypes(include=[np.number])
    X = X.reindex(columns=feature_columns)
    X = X.fillna(impute_values)
    X = X.fillna(0)

    return X


def diagnosticar_modelo():
    print("=" * 80)
    print("DIAGNÓSTICO MODELO V2 - MERVAL")
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
    df = df.sort_values(["Date", "Ticker"])

    print(f"Dataset cargado : {len(df):,} registros")
    print(f"Columnas        : {len(df.columns)}")
    print(f"Desde           : {df['Date'].min().date()}")
    print(f"Hasta           : {df['Date'].max().date()}")
    print(f"Acciones        : {df['Ticker'].nunique()}")

    if "Objetivo_5d" not in df.columns:
        raise ValueError("No existe la columna Objetivo_5d")

    print()
    print("=" * 80)
    print("DISTRIBUCIÓN GENERAL DEL TARGET")
    print("=" * 80)
    print(df["Objetivo_5d"].value_counts(dropna=False))
    print(f"Objetivo positivo general: {df['Objetivo_5d'].mean() * 100:.2f}%")

    ultimas_fechas = sorted(df["Date"].dropna().unique())[-30:]

    resumen_fechas = []

    print()
    print("=" * 80)
    print("DIAGNÓSTICO ÚLTIMAS 30 FECHAS")
    print("=" * 80)

    for fecha in ultimas_fechas:
        df_dia = df[df["Date"] == fecha].copy()

        if df_dia.empty:
            continue

        X = preparar_x(df_dia)

        if hasattr(modelo, "predict_proba"):
            probs = modelo.predict_proba(X)[:, 1]
        else:
            probs = modelo.predict(X)

        df_dia["Prob_IA"] = probs

        resumen_fechas.append({
            "Date": fecha,
            "acciones": df_dia["Ticker"].nunique(),
            "target_positivo_real": df_dia["Objetivo_5d"].mean(),
            "prob_media": df_dia["Prob_IA"].mean(),
            "prob_min": df_dia["Prob_IA"].min(),
            "prob_max": df_dia["Prob_IA"].max(),
            "prob_std": df_dia["Prob_IA"].std(),
            "top_ticker": df_dia.sort_values("Prob_IA", ascending=False).iloc[0]["Ticker"],
            "top_prob": df_dia["Prob_IA"].max(),
        })

    resumen = pd.DataFrame(resumen_fechas)

    print(resumen.to_string(index=False))

    csv_resumen = os.path.join(REPORTES_DIR, "diagnostico_probabilidades_ultimas_30_fechas.csv")
    xlsx_resumen = os.path.join(REPORTES_DIR, "diagnostico_probabilidades_ultimas_30_fechas.xlsx")

    resumen.to_csv(csv_resumen, index=False)
    if exportar_excel_diagnostico_habilitado():
        resumen.to_excel(xlsx_resumen, index=False)

    print()
    print("=" * 80)
    print("ÚLTIMA FECHA - DETALLE COMPLETO")
    print("=" * 80)

    ultima_fecha = df["Date"].max()
    df_ult = df[df["Date"] == ultima_fecha].copy()
    X_ult = preparar_x(df_ult)

    if hasattr(modelo, "predict_proba"):
        probs_ult = modelo.predict_proba(X_ult)[:, 1]
    else:
        probs_ult = modelo.predict(X_ult)

    df_ult["Prob_IA"] = probs_ult

    detalle = df_ult[
        [
            "Date",
            "Ticker",
            "Nombre",
            "Close",
            "Objetivo_5d",
            "Prob_IA",
            "Return_1d",
            "Return_5d",
            "RSI_14",
            "Close_vs_SMA20",
            "Close_vs_SMA50",
            "Close_vs_SMA200",
        ]
    ].sort_values("Prob_IA", ascending=False)

    print(detalle.to_string(index=False))

    csv_detalle = os.path.join(REPORTES_DIR, "diagnostico_ultima_fecha_detalle.csv")
    xlsx_detalle = os.path.join(REPORTES_DIR, "diagnostico_ultima_fecha_detalle.xlsx")

    detalle.to_csv(csv_detalle, index=False)
    if exportar_excel_diagnostico_habilitado():
        detalle.to_excel(xlsx_detalle, index=False)

    print()
    print("=" * 80)
    print("CONTROL DE CALIBRACIÓN GLOBAL")
    print("=" * 80)

    muestra = df.tail(2000).copy()
    X_muestra = preparar_x(muestra)

    if hasattr(modelo, "predict_proba"):
        muestra["Prob_IA"] = modelo.predict_proba(X_muestra)[:, 1]
    else:
        muestra["Prob_IA"] = modelo.predict(X_muestra)

    muestra["bucket"] = pd.cut(
        muestra["Prob_IA"],
        bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        include_lowest=True
    )

    calibracion = muestra.groupby("bucket", observed=True).agg(
        casos=("Objetivo_5d", "count"),
        prob_media=("Prob_IA", "mean"),
        target_real=("Objetivo_5d", "mean")
    ).reset_index()

    print(calibracion.to_string(index=False))

    csv_cal = os.path.join(REPORTES_DIR, "diagnostico_calibracion_global.csv")
    xlsx_cal = os.path.join(REPORTES_DIR, "diagnostico_calibracion_global.xlsx")

    calibracion.to_csv(csv_cal, index=False)
    if exportar_excel_diagnostico_habilitado():
        calibracion.to_excel(xlsx_cal, index=False)

    print()
    print("=" * 80)
    print("ARCHIVOS GENERADOS")
    print("=" * 80)
    print(f"Resumen 30 fechas : {csv_resumen}")
    print(f"Detalle última    : {csv_detalle}")
    print(f"Calibración       : {csv_cal}")
    print(f"Tiempo            : {datetime.now()}")


if __name__ == "__main__":
    diagnosticar_modelo()