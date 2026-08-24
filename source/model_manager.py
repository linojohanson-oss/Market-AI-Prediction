import os
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from source.app_logger import log_print as print

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "datos",
    "merval",
    "procesados",
    "MERVAL_DATASET_IA.csv"
)

MODELOS_DIR = os.path.join(BASE_DIR, "modelos", "merval")
METRICAS_DIR = os.path.join(BASE_DIR, "modelos", "metricas")

os.makedirs(MODELOS_DIR, exist_ok=True)
os.makedirs(METRICAS_DIR, exist_ok=True)


def _parse_bool_env(value, default=False):
    if value is None:
        return default

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def exportar_excel_model_manager_habilitado():
    global_default = _parse_bool_env(os.getenv("MARKET_EXPORT_EXCEL"), default=False)
    return _parse_bool_env(os.getenv("MARKET_MODEL_MANAGER_EXPORT_EXCEL"), default=global_default)


def cargar_modelos():
    modelos = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ),

        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ),

        "GradientBoosting": GradientBoostingClassifier(
            random_state=42
        ),

        "HistGradientBoosting": HistGradientBoostingClassifier(
            random_state=42,
            max_iter=300
        ),

        "AdaBoost": AdaBoostClassifier(
            n_estimators=200,
            random_state=42
        ),

        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            ))
        ]),

        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                max_iter=400,
                random_state=42,
                early_stopping=True
            ))
        ]),
    }

    try:
        from xgboost import XGBClassifier

        modelos["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
    except Exception:
        print("XGBoost no disponible. Se omite.")

    try:
        from lightgbm import LGBMClassifier

        modelos["LightGBM"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=-1,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
            verbose=-1
        )
    except Exception:
        print("LightGBM no disponible. Se omite.")

    try:
        from catboost import CatBoostClassifier

        modelos["CatBoost"] = CatBoostClassifier(
            iterations=300,
            learning_rate=0.03,
            depth=6,
            random_seed=42,
            verbose=False
        )
    except Exception:
        print("CatBoost no disponible. Se omite.")

    return modelos


def detectar_columna_objetivo(df):
    posibles = [
        "Objetivo_5d",
        "objetivo_5d",
        "target",
        "objetivo",
        "Target",
        "sube",
        "Sube",
        "y",
        "target_sube",
        "target_5d",
    ]

    for col in posibles:
        if col in df.columns:
            return col

    raise ValueError(
        "No se encontró columna objetivo. Revisar si se llama Objetivo_5d, target, objetivo o similar."
    )


def preparar_dataset(df):
    target_col = detectar_columna_objetivo(df)

    columnas_excluir = [
        target_col,
        "Date",
        "Fecha",
        "fecha",
        "Ticker",
        "ticker",
        "Nombre",
        "nombre",
        "Mercado",
        "Moneda",
        "Fuente",
        "Activo",
        "activo",
    ]

    features = [
        col for col in df.columns
        if col not in columnas_excluir
    ]

    X = df[features].copy()
    y = df[target_col].copy()

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.select_dtypes(include=[np.number])

    impute_values = X.median(numeric_only=True)
    X = X.fillna(impute_values)
    X = X.fillna(0)

    return X, y, target_col, impute_values


def evaluar_modelo(nombre, modelo, X_train, X_test, y_train, y_test):
    print()
    print("=" * 80)
    print(f"ENTRENANDO MODELO: {nombre}")
    print("=" * 80)

    inicio = datetime.now()

    modelo.fit(X_train, y_train)

    pred = modelo.predict(X_test)

    if hasattr(modelo, "predict_proba"):
        proba = modelo.predict_proba(X_test)[:, 1]
    else:
        proba = pred

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)

    try:
        auc = roc_auc_score(y_test, proba)
    except Exception:
        auc = np.nan

    tiempo = datetime.now() - inicio

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUC ROC  : {auc:.4f}")
    print(f"Tiempo   : {tiempo}")

    modelo_path = os.path.join(MODELOS_DIR, f"{nombre}.pkl")
    joblib.dump(modelo, modelo_path)

    print(f"Modelo guardado: {modelo_path}")

    return {
        "modelo": nombre,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc,
        "tiempo": str(tiempo),
        "archivo_modelo": modelo_path,
    }


def entrenar_model_manager():
    inicio_total = datetime.now()

    print("=" * 80)
    print("MODEL MANAGER 2.0 - MERVAL")
    print("=" * 80)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"No existe el dataset: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    print(f"Dataset cargado : {len(df):,} registros")
    print(f"Columnas totales: {len(df.columns)}")

    X, y, target_col, impute_values = preparar_dataset(df)

    feature_columns_path = os.path.join(MODELOS_DIR, "FEATURE_COLUMNS_MERVAL.pkl")
    impute_values_path = os.path.join(MODELOS_DIR, "IMPUTE_VALUES_MERVAL.pkl")

    joblib.dump(list(X.columns), feature_columns_path)
    joblib.dump(impute_values, impute_values_path)

    print(f"Columnas features guardadas: {feature_columns_path}")
    print(f"Valores imputación guardados: {impute_values_path}")

    print(f"Columna objetivo: {target_col}")
    print(f"Variables usadas: {X.shape[1]}")
    print(f"Objetivo positivo: {y.mean() * 100:.2f}%")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    print(f"Train: {len(X_train):,}")
    print(f"Test : {len(X_test):,}")

    modelos = cargar_modelos()
    resultados = []

    for nombre, modelo in modelos.items():
        try:
            resultado = evaluar_modelo(
                nombre,
                modelo,
                X_train,
                X_test,
                y_train,
                y_test
            )
            resultados.append(resultado)
        except Exception as e:
            print()
            print(f"ERROR entrenando {nombre}: {e}")

    metricas = pd.DataFrame(resultados)

    if metricas.empty:
        raise RuntimeError("No se pudo entrenar ningún modelo.")

    metricas = metricas.sort_values(
        by=["f1", "accuracy", "auc_roc"],
        ascending=False
    )

    print()
    print("=" * 80)
    print("RANKING FINAL DE MODELOS")
    print("=" * 80)
    print(metricas[["modelo", "accuracy", "precision", "recall", "f1", "auc_roc"]])

    csv_path = os.path.join(METRICAS_DIR, "metricas_model_manager_merval.csv")
    xlsx_path = os.path.join(METRICAS_DIR, "metricas_model_manager_merval.xlsx")

    metricas.to_csv(csv_path, index=False)
    excel_resumen = "OMITIDO (usar --excel-all para habilitar)"
    if exportar_excel_model_manager_habilitado():
        metricas.to_excel(xlsx_path, index=False)
        excel_resumen = xlsx_path

    mejor_modelo = metricas.iloc[0]

    best_model_path = os.path.join(MODELOS_DIR, "BEST_MODEL_MERVAL.pkl")
    modelo_ganador = joblib.load(mejor_modelo["archivo_modelo"])
    joblib.dump(modelo_ganador, best_model_path)

    resumen_path = os.path.join(METRICAS_DIR, "best_model_merval.txt")

    with open(resumen_path, "w", encoding="utf-8") as f:
        f.write("BEST MODEL MERVAL\n")
        f.write("=" * 80 + "\n")
        f.write(f"Modelo ganador : {mejor_modelo['modelo']}\n")
        f.write(f"Accuracy       : {mejor_modelo['accuracy']:.4f}\n")
        f.write(f"Precision      : {mejor_modelo['precision']:.4f}\n")
        f.write(f"Recall         : {mejor_modelo['recall']:.4f}\n")
        f.write(f"F1 Score       : {mejor_modelo['f1']:.4f}\n")
        f.write(f"AUC ROC        : {mejor_modelo['auc_roc']:.4f}\n")
        f.write(f"Modelo archivo : {best_model_path}\n")
        f.write(f"Features       : {feature_columns_path}\n")
        f.write(f"Impute values  : {impute_values_path}\n")

    print()
    print("=" * 80)
    print("MEJOR MODELO")
    print("=" * 80)
    print(f"Ganador  : {mejor_modelo['modelo']}")
    print(f"Accuracy : {mejor_modelo['accuracy']:.4f}")
    print(f"F1 Score : {mejor_modelo['f1']:.4f}")
    print(f"AUC ROC  : {mejor_modelo['auc_roc']:.4f}")

    print()
    print(f"Métricas CSV   : {csv_path}")
    print(f"Métricas Excel : {excel_resumen}")
    print(f"Best model     : {best_model_path}")
    print(f"Features       : {feature_columns_path}")
    print(f"Impute values  : {impute_values_path}")
    print(f"Resumen TXT    : {resumen_path}")
    print(f"Tiempo total   : {datetime.now() - inicio_total}")


if __name__ == "__main__":
    entrenar_model_manager()