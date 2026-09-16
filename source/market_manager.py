from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from config import MarketConfig


class MarketManager:
    """
    Capa de compatibilidad y contexto para el viejo Market AI.

    Expone la interfaz esperada por market_macro_report.py:
    - nombre
    - benchmark
    - contexto_macro
    - indicadores_macro
    """

    def __init__(self, nombre_mercado: str = "merval") -> None:
        self.market = MarketConfig(nombre_mercado)

        self.nombre = self.market.display_name
        self.benchmark = self.market.benchmark

        self.contexto_macro = self._cargar_contexto_macro()
        self.indicadores_macro = self._cargar_indicadores_macro()

    def _archivo_contexto(self) -> Path:
        return (
            self.market.processed_dir
            / f"{self.market.market.upper()}_CONTEXTO.csv"
        )

    def _archivo_indicadores(self) -> Path:
        return (
            self.market.processed_dir
            / f"{self.market.market.upper()}_INDICADORES.csv"
        )

    def _cargar_contexto_macro(self) -> Dict[str, Any]:
        archivo = self._archivo_contexto()

        if not archivo.exists():
            return {
                "contexto": "Sin contexto macro disponible.",
                "correlaciones": {},
            }

        df = pd.read_csv(archivo)

        if df.empty:
            return {
                "contexto": "Sin contexto macro disponible.",
                "correlaciones": {},
            }

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(
                df["Date"],
                errors="coerce",
            )
            df.sort_values("Date", inplace=True)

        ultima = df.iloc[-1]

        vix = self._safe_float(
            ultima.get("VIX")
        )
        dxy = self._safe_float(
            ultima.get("DXY")
        )
        nasdaq = self._safe_float(
            ultima.get("NASDAQ")
        )
        sp500 = self._safe_float(
            ultima.get("SP500")
        )

        contexto_texto = (
            f"VIX={vix}, DXY={dxy}, "
            f"NASDAQ={nasdaq}, SP500={sp500}"
        )

        correlaciones = {}

        # Si existen columnas de correlación explícitas, se usan.
        for column in df.columns:
            name = str(column).upper()

            if (
                "CORR" in name
                and "NASDAQ" in name
            ):
                valor = self._safe_float(
                    ultima.get(column)
                )

                if valor is not None:
                    correlaciones["NASDAQ"] = valor
                    break

        return {
            "contexto": contexto_texto,
            "correlaciones": correlaciones,
            "VIX": vix,
            "DXY": dxy,
            "NASDAQ": nasdaq,
            "SP500": sp500,
        }

    def _cargar_indicadores_macro(self) -> Dict[str, Any]:
        archivo = self._archivo_indicadores()

        if not archivo.exists():
            return {}

        df = pd.read_csv(archivo)

        if df.empty:
            return {}

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(
                df["Date"],
                errors="coerce",
            )

        if "Ticker" in df.columns:
            benchmark_ticker = self._resolver_ticker_benchmark(df)

            if benchmark_ticker is not None:
                df = df[
                    df["Ticker"] == benchmark_ticker
                ].copy()

        if df.empty:
            return {}

        df.sort_values(
            "Date",
            inplace=True,
        )

        ultima = df.iloc[-1]

        indicadores = {
            "VIX": self.contexto_macro.get("VIX"),
            "DXY": self.contexto_macro.get("DXY"),
        }

        rv7 = self._safe_float(
            ultima.get("RV_7")
        )

        rv30 = self._safe_float(
            ultima.get("RV_30")
        )

        # Compatibilidad con datasets antiguos que no traen RV_*.
        if rv7 is None:
            rv7 = self._calcular_rv(
                df,
                window=7,
            )

        if rv30 is None:
            rv30 = self._calcular_rv(
                df,
                window=30,
            )

        indicadores["RV_7"] = rv7
        indicadores["RV_30"] = rv30

        return indicadores

    def _resolver_ticker_benchmark(
        self,
        df: pd.DataFrame,
    ) -> str | None:

        benchmark = str(
            self.benchmark
        ).strip()

        tickers = set(
            df["Ticker"]
            .astype(str)
            .unique()
        )

        if benchmark in tickers:
            return benchmark

        # Fallback: primer ticker configurado del mercado
        # que exista en el dataset.
        for ticker in self.market.tickers:
            if ticker in tickers:
                return ticker

        if tickers:
            return sorted(tickers)[0]

        return None

    @staticmethod
    def _calcular_rv(
        df: pd.DataFrame,
        window: int,
    ) -> float | None:

        if "Close" not in df.columns:
            return None

        close = pd.to_numeric(
            df["Close"],
            errors="coerce",
        ).dropna()

        if len(close) <= window:
            return None

        log_ret = np.log(
            close
            / close.shift(1)
        )

        rv = (
            log_ret
            .rolling(window=window)
            .apply(
                lambda x: np.sqrt(
                    (x ** 2).sum()
                ),
                raw=True,
            )
        )

        rv = rv.dropna()

        if rv.empty:
            return None

        return float(
            rv.iloc[-1]
        )

    @staticmethod
    def _safe_float(
        value: Any,
    ) -> float | None:

        try:
            if value is None:
                return None

            value = float(value)

            if np.isnan(value):
                return None

            return value

        except Exception:
            return None


if __name__ == "__main__":
    market = MarketManager("merval")

    print("Mercado    :", market.nombre)
    print("Benchmark  :", market.benchmark)
    print("Contexto   :", market.contexto_macro)
    print("Indicadores:", market.indicadores_macro)
