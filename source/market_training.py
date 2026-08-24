import sys
import os
from pathlib import Path
from datetime import datetime
import warnings

ROOT = Path(__file__).resolve().parent.parent

from source.app_logger import log_print as print

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
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
SAFE_N_JOBS = 1


def _parse_bool_env(value, default=False):
    if value is None:
        return default

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def exportar_excel_training_habilitado():
    global_default = _parse_bool_env(os.getenv("MARKET_EXPORT_EXCEL"), default=False)
    return _parse_bool_env(os.getenv("MARKET_TRAINING_EXPORT_EXCEL"), default=global_default)


def cargar_dataset(market):
    dataset_files = sorted(
        market.processed_dir.glob(f"{market.market.upper()}_DATASET_IA*.csv")
    )

    if not dataset_files:
        raise FileNotFoundError(
            f"No existe dataset IA en {market.processed_dir}. "
            f"Primero ejecutá market_dataset.py"
        )

    dataset_path = dataset_files[-1]

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"No existe dataset IA: {dataset_path}. "
            f"Primero ejecutá market_dataset.py"
        )

    df = pd.read_csv(dataset_path)
    df["Date"] = pd.to_datetime(df["Date"])

    return df


def preparar_datos(df):
    columnas_excluir = [
        "Date",
        "Ticker",
        "Nombre",
        "Mercado",
        "Moneda",
        "Fuente",
        OBJETIVO,
    ]

    features = [c for c in df.columns if c not in columnas_excluir]

    X = df[features].copy()
    y = df[OBJETIVO].copy()

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        shuffle=False,
    )

    return X_train, X_test, y_train, y_test, features


def construir_modelos():
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=14,
        min_samples_leaf=8,
        random_state=42,
        n_jobs=SAFE_N_JOBS,
        class_weight="balanced",
    )

    extra = ExtraTreesClassifier(
        n_estimators=600,
        max_depth=14,
        min_samples_leaf=8,
        random_state=42,
        n_jobs=SAFE_N_JOBS,
        class_weight="balanced",
    )

    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.025,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=42,
        n_jobs=SAFE_N_JOBS,
    )

    lgbm = LGBMClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.025,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=SAFE_N_JOBS,
        class_weight="balanced",
        verbose=-1,
    )

    logistic = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "modelo",
                LogisticRegression(
                    max_iter=2500,
                    class_weight="balanced",
                    n_jobs=SAFE_N_JOBS,
                    random_state=42,
                ),
            ),
        ]
    )

    linear_svc = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "modelo",
                CalibratedClassifierCV(
                    estimator=lgbm,
                    method="sigmoid",
                    cv=5,
                )
            ),
        ]
    )

    modelos = {
        "RandomForest": rf,
        "XGBoost": xgb,
        "LightGBM": lgbm,
        "ExtraTrees": extra,
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            random_state=42,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.03,
            max_leaf_nodes=31,
            random_state=42,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=300,
            learning_rate=0.03,
            random_state=42,
        ),
        "LogisticRegression": logistic,
        "LinearSVC": linear_svc,
        "Voting": VotingClassifier(
            estimators=[
                ("rf", rf),
                ("extra", extra),
                ("xgb", xgb),
                ("lgbm", lgbm),
            ],
            voting="soft",
            n_jobs=-1,
        ),
        "Stacking": StackingClassifier(
            estimators=[
                ("rf", rf),
                ("extra", extra),
                ("xgb", xgb),
                ("lgbm", lgbm),
            ],
            final_estimator=LogisticRegression(
                max_iter=2500,
                class_weight="balanced",
                random_state=42,
            ),
            stack_method="predict_proba",
            n_jobs=SAFE_N_JOBS,
            cv=3,
        ),
    }

    return modelos


def evaluar_modelo(nombre, modelo, X_test, y_test):
    pred = modelo.predict(X_test)

    metricas = {
        "modelo": nombre,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "matriz_confusion": confusion_matrix(y_test, pred).tolist(),
    }

    print()
    print("=" * 70)
    print(f"RESULTADOS - {nombre}")
    print("=" * 70)
    print(f"Accuracy : {metricas['accuracy']:.4f}")
    print(f"Precision: {metricas['precision']:.4f}")
    print(f"Recall   : {metricas['recall']:.4f}")
    print(f"F1 Score : {metricas['f1']:.4f}")
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, pred))
    print()
    print(classification_report(y_test, pred, zero_division=0))

    return metricas


def entrenar_mercado(nombre_mercado="merval"):
    inicio = datetime.now()
    market = MarketConfig(nombre_mercado)

    print("=" * 70)
    print(f"MARKET TRAINING - {market.display_name}")
    print("=" * 70)

    df = cargar_dataset(market)

    print(f"Mercado        : {market.market}")
    print(f"Dataset        : {market.dataset_csv}")
    print(f"Registros      : {len(df):,}")
    print(f"Activos        : {df['Ticker'].nunique()}")
    print(f"Columnas       : {len(df.columns)}")
    print(f"Desde          : {df['Date'].min()}")
    print(f"Hasta          : {df['Date'].max()}")

    X_train, X_test, y_train, y_test, features = preparar_datos(df)

    print()
    print(f"Variables usadas : {len(features)}")
    print(f"Train            : {len(X_train):,}")
    print(f"Test             : {len(X_test):,}")
    print(f"Objetivo positivo train : {y_train.mean():.2%}")
    print(f"Objetivo positivo test  : {y_test.mean():.2%}")

    modelos = construir_modelos()
    resultados = []

    for nombre, modelo in modelos.items():
        print()
        print("=" * 70)
        print(f"Entrenando {nombre}...")
        print("=" * 70)

        try:
            modelo.fit(X_train, y_train)

            metricas = evaluar_modelo(nombre, modelo, X_test, y_test)
            resultados.append(metricas)

            archivo_modelo = market.trained_models_dir / f"{nombre}.pkl"
            joblib.dump(modelo, archivo_modelo)

            print(f"Modelo guardado: {archivo_modelo}")

        except Exception as e:
            print(f"ERROR entrenando {nombre}: {e}")

    joblib.dump(features, market.features_pkl)

    df_metricas = pd.DataFrame(resultados)

    df_metricas["score_modelo"] = (
        df_metricas["f1"] * 0.50 +
        df_metricas["recall"] * 0.25 +
        df_metricas["precision"] * 0.15 +
        df_metricas["accuracy"] * 0.10
    )

    df_metricas.sort_values(
        ["score_modelo", "f1", "accuracy"],
        ascending=False,
        inplace=True,
    )

    metricas_csv = market.metrics_dir / "metricas_modelos.csv"
    metricas_excel = market.metrics_dir / "metricas_modelos.xlsx"

    df_metricas.to_csv(metricas_csv, index=False, encoding="utf-8-sig")
    excel_resumen = "OMITIDO (usar --excel-all para habilitar)"
    if exportar_excel_training_habilitado():
        df_metricas.to_excel(metricas_excel, index=False)
        excel_resumen = str(metricas_excel)

    print()
    print("=" * 70)
    print(f"RESUMEN FINAL MODELOS - {market.display_name}")
    print("=" * 70)

    print(
        df_metricas[
            ["modelo", "accuracy", "precision", "recall", "f1", "score_modelo"]
        ].to_string(index=False)
    )

    print()
    print(f"Métricas CSV   : {metricas_csv}")
    print(f"Métricas Excel : {excel_resumen}")
    print(f"Features PKL   : {market.features_pkl}")
    print(f"Tiempo total   : {datetime.now() - inicio}")

    return df_metricas


if __name__ == "__main__":
    entrenar_mercado("merval")