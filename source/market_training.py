import json
import sys
from pathlib import Path
from datetime import datetime
from threading import Event, Thread
from time import monotonic
import warnings

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import MarketConfig


OBJETIVO = "Objetivo_5d"
TEST_SIZE = 0.25
PURGA_RUEDAS = 5
INTERVALO_PROGRESO_SEGUNDOS = 30


def formatear_duracion(segundos):
    segundos = max(0, int(segundos))
    horas, resto = divmod(segundos, 3600)
    minutos, segundos = divmod(resto, 60)
    if horas:
        return f"{horas:02d}:{minutos:02d}:{segundos:02d}"
    return f"{minutos:02d}:{segundos:02d}"


def ejecutar_con_progreso(descripcion, funcion, intervalo=INTERVALO_PROGRESO_SEGUNDOS):
    """Ejecuta una tarea bloqueante e informa periódicamente que continúa activa."""
    inicio = monotonic()
    detener = Event()

    def mostrar_latido():
        while not detener.wait(intervalo):
            transcurrido = formatear_duracion(monotonic() - inicio)
            print(
                f"  ... {descripcion}: continúa trabajando "
                f"| tiempo transcurrido {transcurrido}",
                flush=True,
            )

    print(f"  INICIO | {descripcion}", flush=True)
    hilo = Thread(target=mostrar_latido, name="market-training-progreso", daemon=True)
    hilo.start()

    try:
        resultado = funcion()
    except Exception:
        transcurrido = formatear_duracion(monotonic() - inicio)
        print(f"  ERROR  | {descripcion} | duración {transcurrido}", flush=True)
        raise
    finally:
        detener.set()
        hilo.join(timeout=1.0)

    transcurrido = formatear_duracion(monotonic() - inicio)
    print(f"  OK     | {descripcion} | duración {transcurrido}", flush=True)
    return resultado


def cargar_dataset(market):
    if not market.dataset_csv.exists():
        raise FileNotFoundError(
            f"No existe dataset IA: {market.dataset_csv}. "
            "Primero ejecutá market_dataset.py"
        )

    df = pd.read_csv(market.dataset_csv)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Ticker", OBJETIVO]).copy()
    df.sort_values(["Date", "Ticker"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def columnas_features(df):
    excluir = {
        "Date", "Ticker", "Nombre", "Mercado", "Moneda", "Fuente", OBJETIVO,
    }
    return [c for c in df.columns if c not in excluir]


def limpiar_X(df, features):
    X = df[features].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)
    return X


def crear_split_temporal(df, test_size=TEST_SIZE, purga_ruedas=PURGA_RUEDAS):
    """Corte por fecha común y purga del horizonte de la etiqueta por ticker."""
    fechas = pd.Index(df["Date"].dropna().sort_values().unique())
    if len(fechas) < 20:
        raise ValueError("No hay suficientes fechas para un corte temporal fiable.")

    posicion = int(len(fechas) * (1.0 - test_size))
    posicion = min(max(posicion, 1), len(fechas) - 1)
    fecha_inicio_test = pd.Timestamp(fechas[posicion])

    train_previo = df[df["Date"] < fecha_inicio_test].copy()
    test = df[df["Date"] >= fecha_inicio_test].copy()

    # Cada Objetivo_5d usa cinco observaciones futuras de su propio ticker.
    # Quitamos las últimas cinco filas pre-test de cada activo para impedir
    # que una etiqueta de entrenamiento alcance fechas pertenecientes al test.
    indices_purga = (
        train_previo.sort_values(["Ticker", "Date"])
        .groupby("Ticker", sort=False)
        .tail(purga_ruedas)
        .index
    )
    train = train_previo.drop(index=indices_purga).copy()

    train.sort_values(["Date", "Ticker"], inplace=True)
    test.sort_values(["Date", "Ticker"], inplace=True)

    if train.empty or test.empty:
        raise ValueError("El corte temporal produjo train o test vacío.")
    if train["Date"].max() >= test["Date"].min():
        raise RuntimeError("Fallo interno: train y test no quedaron separados por fecha.")

    solapadas = set(train["Date"].unique()) & set(test["Date"].unique())
    if solapadas:
        raise RuntimeError("Fallo interno: existen fechas comunes entre train y test.")

    metadata = {
        "estrategia": "fecha_comun_75_25_con_purga_por_ticker",
        "test_size": float(test_size),
        "purga_ruedas": int(purga_ruedas),
        "fecha_inicio_test": fecha_inicio_test.strftime("%Y-%m-%d"),
        "train_fecha_min": train["Date"].min().strftime("%Y-%m-%d"),
        "train_fecha_max": train["Date"].max().strftime("%Y-%m-%d"),
        "test_fecha_min": test["Date"].min().strftime("%Y-%m-%d"),
        "test_fecha_max": test["Date"].max().strftime("%Y-%m-%d"),
        "filas_train": int(len(train)),
        "filas_test": int(len(test)),
        "filas_purgadas": int(len(indices_purga)),
        "tickers_train": int(train["Ticker"].nunique()),
        "tickers_test": int(test["Ticker"].nunique()),
        "tickers_comunes": int(len(set(train["Ticker"]) & set(test["Ticker"]))),
        "fechas_solapadas": 0,
        "separacion_temporal_estricta": True,
    }
    return train, test, metadata


def preparar_datos(df):
    features = columnas_features(df)
    train, test, metadata = crear_split_temporal(df)

    X_train = limpiar_X(train, features)
    X_test = limpiar_X(test, features)
    X_full = limpiar_X(df, features)
    y_train = train[OBJETIVO].astype(int)
    y_test = test[OBJETIVO].astype(int)
    y_full = df[OBJETIVO].astype(int)

    return X_train, X_test, y_train, y_test, X_full, y_full, features, metadata


def construir_modelos():
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=14, min_samples_leaf=8,
        random_state=42, n_jobs=-1, class_weight="balanced",
    )
    extra = ExtraTreesClassifier(
        n_estimators=600, max_depth=14, min_samples_leaf=8,
        random_state=42, n_jobs=-1, class_weight="balanced",
    )
    xgb = XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.025,
        subsample=0.85, colsample_bytree=0.85, eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )
    lgbm = LGBMClassifier(
        n_estimators=500, max_depth=7, learning_rate=0.025,
        subsample=0.85, colsample_bytree=0.85, random_state=42,
        n_jobs=-1, class_weight="balanced", verbose=-1,
    )
    logistic = Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", LogisticRegression(
            max_iter=2500, class_weight="balanced", n_jobs=-1, random_state=42,
        )),
    ])
    linear_svc = Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", CalibratedClassifierCV(
            estimator=LinearSVC(
                class_weight="balanced", random_state=42, max_iter=10000,
            ),
            method="sigmoid", cv=5,
        )),
    ])

    return {
        "RandomForest": rf,
        "XGBoost": xgb,
        "LightGBM": lgbm,
        "ExtraTrees": extra,
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.03, max_depth=3, random_state=42,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.03, max_leaf_nodes=31, random_state=42,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=300, learning_rate=0.03, random_state=42,
        ),
        "LogisticRegression": logistic,
        "LinearSVC": linear_svc,
        "Voting": VotingClassifier(
            estimators=[("rf", rf), ("extra", extra), ("xgb", xgb), ("lgbm", lgbm)],
            voting="soft", n_jobs=-1,
        ),
        "Stacking": StackingClassifier(
            estimators=[("rf", rf), ("extra", extra), ("xgb", xgb), ("lgbm", lgbm)],
            final_estimator=LogisticRegression(
                max_iter=2500, class_weight="balanced", random_state=42,
            ),
            stack_method="predict_proba", n_jobs=-1, cv=3,
        ),
    }


def probabilidades_modelo(modelo, X):
    if hasattr(modelo, "predict_proba"):
        prob = np.asarray(modelo.predict_proba(X))[:, 1]
    elif hasattr(modelo, "decision_function"):
        margen = np.clip(np.asarray(modelo.decision_function(X)), -50.0, 50.0)
        prob = 1.0 / (1.0 + np.exp(-margen))
    else:
        prob = np.asarray(modelo.predict(X), dtype=float)
    return np.clip(prob.astype(float).reshape(-1), 0.0, 1.0)


def expected_calibration_error(y_true, prob, bins=10):
    tabla = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(prob)})
    tabla["bin"] = pd.cut(
        tabla["p"], bins=np.linspace(0.0, 1.0, bins + 1),
        include_lowest=True, duplicates="drop",
    )
    resumen = tabla.groupby("bin", observed=True).agg(
        filas=("y", "size"), prob_media=("p", "mean"), frecuencia_real=("y", "mean")
    )
    if resumen.empty:
        return float("nan")
    return float(
        (resumen["filas"] * (resumen["prob_media"] - resumen["frecuencia_real"]).abs()).sum()
        / resumen["filas"].sum()
    )


def evaluar_modelo(nombre, modelo, X_test, y_test, metadata):
    pred = modelo.predict(X_test)
    prob = probabilidades_modelo(modelo, X_test)

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    balanced_accuracy = balanced_accuracy_score(y_test, pred)
    mcc = matthews_corrcoef(y_test, pred)
    positive_rate_true = float(np.mean(y_test))
    positive_rate_pred = float(np.mean(pred))

    mcc_normalizado = max(0.0, min(1.0, mcc))
    if positive_rate_pred < 0.20:
        penalizacion_clase = max(0.25, positive_rate_pred / 0.20)
    elif positive_rate_pred > 0.80:
        penalizacion_clase = max(0.25, (1.0 - positive_rate_pred) / 0.20)
    else:
        penalizacion_clase = 1.0

    score_base = (
        balanced_accuracy * 0.30 + mcc_normalizado * 0.25
        + f1 * 0.20 + precision * 0.15 + accuracy * 0.10
    )
    score_modelo = score_base * penalizacion_clase

    try:
        auc = roc_auc_score(y_test, prob)
    except ValueError:
        auc = float("nan")

    metricas = {
        "modelo": nombre,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
        "positive_rate_true": positive_rate_true,
        "positive_rate_pred": positive_rate_pred,
        "probability_mean": float(np.mean(prob)),
        "brier_score": brier_score_loss(y_test, prob),
        "log_loss": log_loss(y_test, prob, labels=[0, 1]),
        "roc_auc": auc,
        "ece_10": expected_calibration_error(y_test, prob),
        "penalizacion_clase": penalizacion_clase,
        "score_base": score_base,
        "score_modelo": score_modelo,
        "matriz_confusion": confusion_matrix(y_test, pred).tolist(),
        "fecha_inicio_test": metadata["fecha_inicio_test"],
        "filas_train": metadata["filas_train"],
        "filas_test": metadata["filas_test"],
        "purga_ruedas": metadata["purga_ruedas"],
        "separacion_temporal_estricta": True,
    }

    print()
    print("=" * 70)
    print(f"RESULTADOS TEMPORALES - {nombre}")
    print("=" * 70)
    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Precision         : {precision:.4f}")
    print(f"Recall            : {recall:.4f}")
    print(f"F1 Score          : {f1:.4f}")
    print(f"Balanced Accuracy : {balanced_accuracy:.4f}")
    print(f"MCC               : {mcc:.4f}")
    print(f"Brier Score       : {metricas['brier_score']:.4f}")
    print(f"Log Loss          : {metricas['log_loss']:.4f}")
    print(f"ROC AUC           : {auc:.4f}")
    print(f"ECE 10            : {metricas['ece_10']:.4f}")
    print(f"Positivos reales  : {positive_rate_true:.2%}")
    print(f"Positivos pred.   : {positive_rate_pred:.2%}")
    print(f"Penalización      : {penalizacion_clase:.4f}")
    print(f"Score Modelo V2   : {score_modelo:.4f}")
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, pred))
    print()
    print(classification_report(y_test, pred, zero_division=0))
    return metricas


def entrenar_mercado(nombre_mercado="merval"):
    inicio = datetime.now()
    market = MarketConfig(nombre_mercado)

    print("=" * 86)
    print(f"MARKET TRAINING TEMPORAL V3 - {market.display_name}")
    print("=" * 86)

    df = cargar_dataset(market)
    X_train, X_test, y_train, y_test, X_full, y_full, features, meta = preparar_datos(df)

    print(f"Mercado                 : {market.market}")
    print(f"Dataset                 : {market.dataset_csv}")
    print(f"Registros válidos       : {len(df):,}")
    print(f"Activos                 : {df['Ticker'].nunique()}")
    print(f"Variables               : {len(features)}")
    print(f"Train temporal          : {len(X_train):,}")
    print(f"Test temporal           : {len(X_test):,}")
    print(f"Filas purgadas          : {meta['filas_purgadas']:,}")
    print(f"Train fechas            : {meta['train_fecha_min']} -> {meta['train_fecha_max']}")
    print(f"Test fechas             : {meta['test_fecha_min']} -> {meta['test_fecha_max']}")
    print(f"Fechas solapadas        : {meta['fechas_solapadas']}")
    print(f"Tickers train/test      : {meta['tickers_train']}/{meta['tickers_test']}")
    print(f"Objetivo positivo train : {y_train.mean():.2%}")
    print(f"Objetivo positivo test  : {y_test.mean():.2%}")

    validacion_dir = market.trained_models_dir / "validacion_temporal"
    validacion_dir.mkdir(parents=True, exist_ok=True)
    market.trained_models_dir.mkdir(parents=True, exist_ok=True)
    market.metrics_dir.mkdir(parents=True, exist_ok=True)

    modelos = construir_modelos()
    resultados = []

    total_modelos = len(modelos)

    for numero_modelo, (nombre, plantilla) in enumerate(modelos.items(), start=1):
        print()
        print("=" * 86)
        print(f"MODELO {numero_modelo}/{total_modelos} - {nombre}")
        print(f"[1/2] VALIDACIÓN TEMPORAL - {nombre}")
        print("=" * 86)
        try:
            modelo_validacion = clone(plantilla)
            ejecutar_con_progreso(
                f"{nombre} - ajuste de validación temporal",
                lambda: modelo_validacion.fit(X_train, y_train),
            )
            metricas = evaluar_modelo(
                nombre, modelo_validacion, X_test, y_test, meta
            )
            resultados.append(metricas)
            joblib.dump(modelo_validacion, validacion_dir / f"{nombre}.pkl")

            print()
            print(f"[2/2] REENTRENAMIENTO FULL - {nombre}")
            modelo_full = clone(plantilla)
            ejecutar_con_progreso(
                f"{nombre} - reentrenamiento FULL",
                lambda: modelo_full.fit(X_full, y_full),
            )
            archivo_modelo = market.trained_models_dir / f"{nombre}.pkl"
            joblib.dump(modelo_full, archivo_modelo)
            print(f"Modelo FULL guardado: {archivo_modelo}")
        except Exception as exc:
            print(f"ERROR entrenando {nombre}: {exc}")

    if not resultados:
        raise RuntimeError("No se pudo entrenar ningún modelo.")

    joblib.dump(features, market.features_pkl)
    df_metricas = pd.DataFrame(resultados).sort_values(
        ["score_modelo", "balanced_accuracy", "mcc", "f1", "accuracy"],
        ascending=False,
    )

    metricas_csv = market.metrics_dir / "metricas_modelos.csv"
    metricas_excel = market.metrics_dir / "metricas_modelos.xlsx"
    split_json = market.metrics_dir / "split_temporal.json"
    df_metricas.to_csv(metricas_csv, index=False, encoding="utf-8-sig")
    df_metricas.to_excel(metricas_excel, index=False)
    with open(split_json, "w", encoding="utf-8") as archivo:
        json.dump(meta, archivo, indent=4, ensure_ascii=False)

    print()
    print("=" * 86)
    print(f"RESUMEN FINAL TEMPORAL - {market.display_name}")
    print("=" * 86)
    print(df_metricas[[
        "modelo", "accuracy", "balanced_accuracy", "mcc", "f1",
        "brier_score", "log_loss", "roc_auc", "ece_10", "score_modelo",
    ]].to_string(index=False))
    print()
    print(f"Métricas CSV        : {metricas_csv}")
    print(f"Métricas Excel      : {metricas_excel}")
    print(f"Split temporal      : {split_json}")
    print(f"Modelos validación  : {validacion_dir}")
    print(f"Modelos FULL        : {market.trained_models_dir}")
    print(f"Features PKL        : {market.features_pkl}")
    print(f"Tiempo total        : {datetime.now() - inicio}")
    return df_metricas


if __name__ == "__main__":
    entrenar_mercado("merval")
