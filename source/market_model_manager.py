import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from config import MarketConfig


PESO_MINIMO = 0.01
AUC_MINIMO = 0.515
BALANCED_ACCURACY_MINIMA = 0.50
ECE_MAXIMO = 0.15


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
        obligatorias = ["modelo", "accuracy", "precision", "recall", "f1"]
        faltantes = [c for c in obligatorias if c not in df.columns]
        if faltantes:
            raise ValueError(f"Faltan columnas obligatorias: {faltantes}")

        numericas = [
            "accuracy", "precision", "recall", "f1", "balanced_accuracy",
            "mcc", "positive_rate_true", "positive_rate_pred",
            "probability_mean", "brier_score", "log_loss", "roc_auc",
            "ece_10", "penalizacion_clase", "score_base", "score_modelo",
        ]
        for columna in numericas:
            if columna in df.columns:
                df[columna] = pd.to_numeric(df[columna], errors="coerce")

        df.dropna(subset=["modelo", "accuracy", "precision", "recall", "f1"], inplace=True)
        if df.empty:
            raise ValueError("El archivo de métricas no contiene modelos válidos.")
        return df

    @staticmethod
    def _penalizacion_colapso(df):
        if "penalizacion_clase" in df.columns:
            return df["penalizacion_clase"].fillna(1.0).clip(0.25, 1.0)

        tasa = df["positive_rate_pred"].fillna(0.5).clip(0.0, 1.0)
        penalizacion = pd.Series(1.0, index=df.index, dtype=float)
        baja = tasa < 0.20
        alta = tasa > 0.80
        penalizacion.loc[baja] = (tasa.loc[baja] / 0.20).clip(lower=0.25)
        penalizacion.loc[alta] = ((1.0 - tasa.loc[alta]) / 0.20).clip(lower=0.25)
        return penalizacion

    def _calcular_score_modelo(self):
        df = self.metricas.copy()
        columnas_v3 = {
            "balanced_accuracy", "mcc", "positive_rate_true",
            "positive_rate_pred", "brier_score", "roc_auc", "ece_10",
        }

        if columnas_v3.issubset(df.columns):
            df["version_score"] = "V3_PROBABILISTICO_TEMPORAL"

            tasa_real = df["positive_rate_true"].clip(0.001, 0.999)
            brier_referencia = tasa_real * (1.0 - tasa_real)
            df["brier_referencia"] = brier_referencia
            df["brier_skill"] = (
                1.0 - df["brier_score"] / brier_referencia
            ).clip(0.0, 1.0)
            df["auc_skill"] = ((df["roc_auc"] - 0.50) / 0.50).clip(0.0, 1.0)
            df["balanced_skill"] = (
                (df["balanced_accuracy"] - 0.50) / 0.50
            ).clip(0.0, 1.0)
            df["mcc_skill"] = df["mcc"].clip(0.0, 1.0)

            # La calibración reduce el score, pero no puede crear capacidad
            # predictiva por sí sola. Un modelo casi constante puede tener ECE
            # bajo y, aun así, no discriminar activos ganadores de perdedores.
            df["factor_calibracion"] = (
                1.0 - df["ece_10"] / 0.25
            ).clip(0.25, 1.0)
            df["penalizacion_clase"] = self._penalizacion_colapso(df)

            df["score_base_v3"] = (
                df["auc_skill"] * 0.40
                + df["balanced_skill"] * 0.25
                + df["mcc_skill"] * 0.20
                + df["brier_skill"] * 0.15
            )
            df["score_modelo"] = (
                df["score_base_v3"]
                * df["factor_calibracion"]
                * df["penalizacion_clase"]
            )

            separacion_ok = pd.Series(True, index=df.index)
            if "separacion_temporal_estricta" in df.columns:
                separacion_ok = (
                    df["separacion_temporal_estricta"]
                    .astype(str).str.lower().isin(["true", "1", "si", "sí"])
                )

            df["elegible"] = (
                separacion_ok
                & (df["roc_auc"] >= AUC_MINIMO)
                & (df["balanced_accuracy"] >= BALANCED_ACCURACY_MINIMA)
                & (df["mcc"] > 0.0)
                & (df["ece_10"] <= ECE_MAXIMO)
                & df["positive_rate_pred"].between(0.20, 0.80)
                & (df["brier_score"] <= brier_referencia * 1.10)
            )

            def motivo(row):
                razones = []
                if not bool(separacion_ok.loc[row.name]):
                    razones.append("sin separación temporal")
                if row["roc_auc"] < AUC_MINIMO:
                    razones.append(f"AUC<{AUC_MINIMO:.3f}")
                if row["balanced_accuracy"] < BALANCED_ACCURACY_MINIMA:
                    razones.append("balanced_accuracy<0.50")
                if row["mcc"] <= 0:
                    razones.append("MCC<=0")
                if row["ece_10"] > ECE_MAXIMO:
                    razones.append(f"ECE>{ECE_MAXIMO:.2f}")
                if not 0.20 <= row["positive_rate_pred"] <= 0.80:
                    razones.append("colapso de clase")
                if row["brier_score"] > row["brier_referencia"] * 1.10:
                    razones.append("Brier excesivo")
                return "ELEGIBLE" if not razones else "; ".join(razones)

            df["motivo_elegibilidad"] = df.apply(motivo, axis=1)
            orden = ["elegible", "score_modelo", "roc_auc", "brier_score"]
            ascendente = [False, False, False, True]
        else:
            # Compatibilidad para mercados todavía no reentrenados con V3.
            df["version_score"] = "V2_COMPATIBILIDAD"
            if {"balanced_accuracy", "mcc", "positive_rate_pred"}.issubset(df.columns):
                df["penalizacion_clase"] = self._penalizacion_colapso(df)
                df["score_modelo"] = (
                    df["balanced_accuracy"] * 0.30
                    + df["mcc"].clip(0.0, 1.0) * 0.25
                    + df["f1"] * 0.20
                    + df["precision"] * 0.15
                    + df["accuracy"] * 0.10
                ) * df["penalizacion_clase"]
                df["elegible"] = df["penalizacion_clase"] >= 0.50
                df["motivo_elegibilidad"] = np.where(
                    df["elegible"], "ELEGIBLE V2", "colapso de clase V2"
                )
                orden = ["elegible", "score_modelo", "balanced_accuracy", "mcc"]
                ascendente = [False, False, False, False]
            else:
                df["score_modelo"] = (
                    df["f1"] * 0.50 + df["recall"] * 0.25
                    + df["precision"] * 0.15 + df["accuracy"] * 0.10
                )
                df["elegible"] = True
                df["motivo_elegibilidad"] = "ELEGIBLE LEGACY"
                orden = ["score_modelo", "f1", "accuracy"]
                ascendente = [False, False, False]

        df.sort_values(orden, ascending=ascendente, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["ranking"] = df.index + 1
        return df

    def _buscar_mejor_modelo(self):
        elegibles = self.metricas[self.metricas["elegible"]]
        if elegibles.empty:
            raise RuntimeError(
                "Ningún modelo supera los controles mínimos de calidad. "
                "No se generará scoring con modelos no fiables."
            )
        return elegibles.iloc[0]

    def _crear_info_modelo_activo(self):
        row = self.mejor_modelo
        info = {
            "mercado": self.market.market,
            "display_name": self.market.display_name,
            "modelo": row["modelo"],
            "ranking": int(row["ranking"]),
            "version_score": row["version_score"],
            "accuracy": float(row["accuracy"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1"]),
            "score_modelo": float(row["score_modelo"]),
            "archivo_modelo": str(self.market.trained_models_dir / f"{row['modelo']}.pkl"),
            "archivo_features": str(self.market.features_pkl),
            "archivo_dataset": str(self.market.dataset_csv),
            "fecha_actualizacion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        for col in ["balanced_accuracy", "mcc", "brier_score", "roc_auc", "ece_10"]:
            if col in row.index and pd.notna(row[col]):
                info[col] = float(row[col])
        return info

    def _crear_ensemble(self):
        df = self.metricas.copy()
        elegibles = df[df["elegible"]].copy()
        if elegibles.empty:
            raise RuntimeError("No existen modelos elegibles para el ensemble.")

        total_score = float(elegibles["score_modelo"].sum())
        if total_score <= 0:
            raise RuntimeError("Los modelos elegibles no poseen score predictivo positivo.")

        modelos = []
        for _, row in df.iterrows():
            participa = bool(row["elegible"])
            peso = float(row["score_modelo"] / total_score) if participa else 0.0
            participa = participa and peso >= PESO_MINIMO

            item = {
                "modelo": row["modelo"],
                "peso": peso if participa else 0.0,
                "peso_pct": (peso * 100.0) if participa else 0.0,
                "ranking": int(row["ranking"]),
                "version_score": row["version_score"],
                "score_modelo": float(row["score_modelo"]),
                "participa": participa,
                "motivo": row["motivo_elegibilidad"],
                "archivo_modelo": str(self.market.trained_models_dir / f"{row['modelo']}.pkl"),
                "archivo_features": str(self.market.features_pkl),
            }
            for col in [
                "accuracy", "precision", "recall", "f1", "balanced_accuracy",
                "mcc", "brier_score", "roc_auc", "ece_10",
            ]:
                if col in row.index and pd.notna(row[col]):
                    item[col] = float(row[col])
            modelos.append(item)

        return {
            "mercado": self.market.market,
            "display_name": self.market.display_name,
            "fecha_actualizacion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "criterio": (
                "Score V3 probabilístico temporal: 40% AUC skill + 25% balanced "
                "accuracy skill + 20% MCC positivo + 15% Brier skill; penalizado "
                "por calibración y colapso de clase"
            ),
            "modelos_elegibles": int(sum(m["participa"] for m in modelos)),
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
        preferidas = [
            "ranking", "modelo", "version_score", "elegible",
            "motivo_elegibilidad", "score_modelo", "roc_auc", "brier_score",
            "ece_10", "balanced_accuracy", "mcc", "accuracy", "precision",
            "recall", "f1", "positive_rate_true", "positive_rate_pred",
        ]
        columnas = [c for c in preferidas if c in self.metricas.columns]
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
        """Carga exclusivamente modelos autorizados por el ranking V3."""
        modelos = []
        for item in self.ensemble["modelos"]:
            if not item.get("participa", False):
                continue
            archivo = Path(item["archivo_modelo"])
            if not archivo.exists():
                print(f"Modelo elegible no encontrado: {archivo}")
                continue
            modelos.append({
                "modelo": item["modelo"],
                "peso": item["peso"],
                "participa": True,
                "objeto": joblib.load(archivo),
            })
        if not modelos:
            raise RuntimeError("No se pudo cargar ningún modelo elegible del ensemble.")
        return modelos

    def resumen(self):
        activo = self.guardar_modelo_activo()
        ensemble = self.guardar_ensemble()
        ranking_csv, ranking_excel = self.guardar_ranking_modelos()

        print("=" * 96)
        print(f"MARKET MODEL MANAGER V3 PROBABILÍSTICO - {self.market.display_name}")
        print("=" * 96)
        print(f"Modelo activo : {self.info_activa['modelo']}")
        print(f"Score V3      : {self.info_activa['score_modelo']:.6f}")
        if "roc_auc" in self.info_activa:
            print(f"ROC AUC       : {self.info_activa['roc_auc']:.4f}")
            print(f"Brier         : {self.info_activa['brier_score']:.4f}")
            print(f"ECE 10        : {self.info_activa['ece_10']:.4f}")

        print()
        print("MODELOS Y ELEGIBILIDAD")
        print("-" * 96)
        for m in self.ensemble["modelos"]:
            auc = m.get("roc_auc", float("nan"))
            print(
                f"{m['modelo']:24} Ranking: {m['ranking']:2d} | "
                f"AUC: {auc:.4f} | Peso: {m['peso_pct']:6.2f}% | "
                f"Participa: {str(m['participa']):5} | {m['motivo']}"
            )

        print()
        print(f"Modelo activo JSON : {activo}")
        print(f"Ensemble JSON      : {ensemble}")
        print(f"Ranking CSV        : {ranking_csv}")
        print(f"Ranking Excel      : {ranking_excel}")


def actualizar_market_model_manager(nombre_mercado="merval"):
    manager = MarketModelManager(nombre_mercado)
    manager.resumen()
    return manager


if __name__ == "__main__":
    actualizar_market_model_manager("merval")
