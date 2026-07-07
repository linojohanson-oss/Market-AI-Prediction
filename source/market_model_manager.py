import sys
from pathlib import Path
from datetime import datetime
import json

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import pandas as pd
import joblib

from config import MarketConfig


F1_MINIMO_ACTIVO = 0.0
PESO_MINIMO = 0.01


class MarketModelManager:

    def __init__(self, nombre_mercado="merval"):
        self.market = MarketConfig(nombre_mercado)
        self.metricas = self._cargar_metricas()
        self.metricas = self._calcular_score_modelo()
        self.mejor_modelo = self._buscar_mejor_modelo()
        self.ensemble = self._crear_ensemble()
        self.info_activa = self._crear_info_modelo_activo()

    def _cargar_metricas(self):
        archivo = self.market.metrics_dir / "metricas_modelos.csv"

        if not archivo.exists():
            raise FileNotFoundError(f"No existe archivo de métricas: {archivo}")

        df = pd.read_csv(archivo)

        columnas = ["accuracy", "precision", "recall", "f1"]

        for col in columnas:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.dropna(subset=columnas, inplace=True)

        return df

    def _calcular_score_modelo(self):
        df = self.metricas.copy()

        df["score_modelo"] = (
            df["f1"] * 0.50 +
            df["recall"] * 0.25 +
            df["precision"] * 0.15 +
            df["accuracy"] * 0.10
        )

        df.sort_values(
            ["score_modelo", "f1", "accuracy"],
            ascending=False,
            inplace=True,
        )

        df.reset_index(drop=True, inplace=True)
        df["ranking"] = df.index + 1

        return df

    def _buscar_mejor_modelo(self):
        return self.metricas.iloc[0]

    def _crear_info_modelo_activo(self):
        nombre = self.mejor_modelo["modelo"]

        return {
            "mercado": self.market.market,
            "display_name": self.market.display_name,
            "modelo": nombre,
            "ranking": int(self.mejor_modelo["ranking"]),
            "accuracy": float(self.mejor_modelo["accuracy"]),
            "precision": float(self.mejor_modelo["precision"]),
            "recall": float(self.mejor_modelo["recall"]),
            "f1": float(self.mejor_modelo["f1"]),
            "score_modelo": float(self.mejor_modelo["score_modelo"]),
            "archivo_modelo": str(self.market.trained_models_dir / f"{nombre}.pkl"),
            "archivo_features": str(self.market.features_pkl),
            "archivo_dataset": str(self.market.dataset_csv),
            "fecha_actualizacion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _crear_ensemble(self):
        df = self.metricas.copy()

        # Todos los modelos participan, salvo que se defina un mínimo
        df = df[df["f1"] >= F1_MINIMO_ACTIVO].copy()

        if df.empty:
            df = self.metricas.copy()

        total_score = df["score_modelo"].sum()

        modelos = []

        for _, row in df.iterrows():
            nombre = row["modelo"]

            if total_score > 0:
                peso = float(row["score_modelo"] / total_score)
            else:
                peso = 1 / len(df)

            # No eliminamos modelos; solo registramos pesos bajos
            modelos.append({
                "modelo": nombre,
                "peso": peso,
                "peso_pct": peso * 100,
                "ranking": int(row["ranking"]),
                "accuracy": float(row["accuracy"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
                "score_modelo": float(row["score_modelo"]),
                "participa": bool(peso >= PESO_MINIMO),
                "archivo_modelo": str(self.market.trained_models_dir / f"{nombre}.pkl"),
                "archivo_features": str(self.market.features_pkl),
            })

        return {
            "mercado": self.market.market,
            "display_name": self.market.display_name,
            "fecha_actualizacion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "criterio": "score_modelo = 50% F1 + 25% Recall + 15% Precision + 10% Accuracy",
            "modelos": modelos,
        }

    def guardar_modelo_activo(self):
        archivo = self.market.model_active_json

        with open(archivo, "w", encoding="utf-8") as f:
            json.dump(self.info_activa, f, indent=4, ensure_ascii=False)

        return archivo

    def guardar_ensemble(self):
        archivo = self.market.ensemble_json

        with open(archivo, "w", encoding="utf-8") as f:
            json.dump(self.ensemble, f, indent=4, ensure_ascii=False)

        return archivo

    def guardar_ranking_modelos(self):
        archivo_csv = self.market.ranking_models_csv
        archivo_excel = self.market.ranking_models_excel

        columnas = [
            "ranking",
            "modelo",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "score_modelo",
        ]

        ranking = self.metricas[columnas].copy()

        ranking.to_csv(archivo_csv, index=False, encoding="utf-8-sig")
        ranking.to_excel(archivo_excel, index=False)

        return archivo_csv, archivo_excel

    def cargar_modelo_activo(self):
        archivo = Path(self.info_activa["archivo_modelo"])

        if not archivo.exists():
            raise FileNotFoundError(f"No existe modelo activo: {archivo}")

        return joblib.load(archivo)

    def cargar_features(self):
        archivo = Path(self.info_activa["archivo_features"])

        if not archivo.exists():
            raise FileNotFoundError(f"No existe archivo features: {archivo}")

        return joblib.load(archivo)

    def cargar_dataset(self):
        archivo = Path(self.info_activa["archivo_dataset"])

        if not archivo.exists():
            raise FileNotFoundError(f"No existe dataset: {archivo}")

        df = pd.read_csv(archivo)
        df["Date"] = pd.to_datetime(df["Date"])

        return df

    def cargar_modelos_ensemble(self):
        modelos = []

        for item in self.ensemble["modelos"]:
            archivo = Path(item["archivo_modelo"])

            if not archivo.exists():
                print(f"Modelo no encontrado: {archivo}")
                continue

            modelos.append({
                "modelo": item["modelo"],
                "peso": item["peso"],
                "participa": item["participa"],
                "objeto": joblib.load(archivo),
            })

        return modelos

    def resumen(self):
        archivo_activo = self.guardar_modelo_activo()
        archivo_ensemble = self.guardar_ensemble()
        ranking_csv, ranking_excel = self.guardar_ranking_modelos()

        print("=" * 70)
        print(f"MARKET MODEL MANAGER - {self.market.display_name}")
        print("=" * 70)

        print()
        print("Modelo activo individual")
        print("-" * 70)
        print(f"Modelo       : {self.info_activa['modelo']}")
        print(f"Ranking      : {self.info_activa['ranking']}")
        print(f"Accuracy     : {self.info_activa['accuracy']:.4f}")
        print(f"Precision    : {self.info_activa['precision']:.4f}")
        print(f"Recall       : {self.info_activa['recall']:.4f}")
        print(f"F1           : {self.info_activa['f1']:.4f}")
        print(f"Score Modelo : {self.info_activa['score_modelo']:.4f}")

        print()
        print("Ensemble dinámico")
        print("-" * 70)

        for m in self.ensemble["modelos"]:
            print(
                f"{m['modelo']:24} "
                f"Peso: {m['peso_pct']:6.2f}% | "
                f"F1: {m['f1']:.4f} | "
                f"Recall: {m['recall']:.4f} | "
                f"Participa: {m['participa']}"
            )

        print()
        print("Archivos generados")
        print("-" * 70)
        print(f"Modelo activo JSON : {archivo_activo}")
        print(f"Ensemble JSON      : {archivo_ensemble}")
        print(f"Ranking CSV        : {ranking_csv}")
        print(f"Ranking Excel      : {ranking_excel}")

        print()
        print("Ranking modelos")
        print("-" * 70)
        print(
            self.metricas[
                ["ranking", "modelo", "accuracy", "precision", "recall", "f1", "score_modelo"]
            ].to_string(index=False)
        )


def actualizar_market_model_manager(nombre_mercado="merval"):
    manager = MarketModelManager(nombre_mercado)
    manager.resumen()
    return manager


if __name__ == "__main__":
    actualizar_market_model_manager("merval")