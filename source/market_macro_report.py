from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from source.market_manager import MarketManager


# =============================================================================
# UTILIDADES
# =============================================================================

def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None

        value = float(value)

        if not np.isfinite(value):
            return None

        return value

    except Exception:
        return None


def _fmt_number(
    value: Any,
    decimals: int = 2,
    suffix: str = "",
) -> str:
    value = _safe_float(value)

    if value is None:
        return "N/A"

    return f"{value:.{decimals}f}{suffix}"


def _fmt_pct(
    value: Any,
    decimals: int = 2,
) -> str:
    value = _safe_float(value)

    if value is None:
        return "N/A"

    return f"{value * 100:.{decimals}f}%"


def _latest_value(
    row: pd.Series,
    column: str,
) -> Optional[float]:
    if column not in row.index:
        return None

    return _safe_float(
        row.get(column)
    )


# =============================================================================
# LECTURA DE ARCHIVOS
# =============================================================================

def _load_context(
    manager: MarketManager,
) -> pd.DataFrame:

    archivo = (
        manager.market.processed_dir
        / f"{manager.market.market.upper()}_CONTEXTO.csv"
    )

    if not archivo.exists():
        return pd.DataFrame()

    df = pd.read_csv(archivo)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce",
        )

        df.dropna(
            subset=["Date"],
            inplace=True,
        )

        df.sort_values(
            "Date",
            inplace=True,
        )

    return df


def _load_indicators(
    manager: MarketManager,
) -> pd.DataFrame:

    archivo = (
        manager.market.processed_dir
        / f"{manager.market.market.upper()}_INDICADORES.csv"
    )

    if not archivo.exists():
        return pd.DataFrame()

    df = pd.read_csv(archivo)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce",
        )

        df.dropna(
            subset=["Date"],
            inplace=True,
        )

    return df


def _benchmark_frame(
    manager: MarketManager,
    indicadores: pd.DataFrame,
) -> pd.DataFrame:

    if indicadores.empty:
        return pd.DataFrame()

    if "Ticker" not in indicadores.columns:
        return indicadores.copy()

    tickers = set(
        indicadores["Ticker"]
        .astype(str)
        .unique()
    )

    benchmark = str(
        manager.benchmark
    )

    if benchmark in tickers:
        ticker = benchmark

    else:
        ticker = None

        for candidate in manager.market.tickers:
            if candidate in tickers:
                ticker = candidate
                break

        if ticker is None and tickers:
            ticker = sorted(tickers)[0]

    if ticker is None:
        return pd.DataFrame()

    result = indicadores[
        indicadores["Ticker"]
        .astype(str)
        == ticker
    ].copy()

    result.sort_values(
        "Date",
        inplace=True,
    )

    return result


# =============================================================================
# CORRELACIONES
# =============================================================================

def _rolling_correlation(
    benchmark_df: pd.DataFrame,
    contexto: pd.DataFrame,
    macro_column: str,
    window: int,
) -> Optional[float]:

    if (
        benchmark_df.empty
        or contexto.empty
        or "Date" not in benchmark_df.columns
        or "Close" not in benchmark_df.columns
        or "Date" not in contexto.columns
        or macro_column not in contexto.columns
    ):
        return None

    left = benchmark_df[
        ["Date", "Close"]
    ].copy()

    right = contexto[
        ["Date", macro_column]
    ].copy()

    left["Close"] = pd.to_numeric(
        left["Close"],
        errors="coerce",
    )

    right[macro_column] = pd.to_numeric(
        right[macro_column],
        errors="coerce",
    )

    merged = pd.merge(
        left,
        right,
        on="Date",
        how="inner",
    ).dropna()

    if len(merged) < window + 2:
        return None

    merged["benchmark_return"] = (
        merged["Close"]
        .pct_change()
    )

    merged["macro_return"] = (
        merged[macro_column]
        .pct_change()
    )

    sample = (
        merged[
            [
                "benchmark_return",
                "macro_return",
            ]
        ]
        .dropna()
        .tail(window)
    )

    if len(sample) < max(10, window // 2):
        return None

    correlation = (
        sample["benchmark_return"]
        .corr(
            sample["macro_return"]
        )
    )

    return _safe_float(
        correlation
    )


# =============================================================================
# INTERPRETACIONES
# =============================================================================

def _interpret_vix(vix: Optional[float]) -> str:

    if vix is None:
        return "Sin lectura disponible."

    if vix < 15:
        return (
            "Volatilidad implícita baja. "
            "El mercado global muestra elevada calma."
        )

    if vix < 20:
        return (
            "Volatilidad implícita moderada. "
            "No se observa estrés global significativo."
        )

    if vix < 30:
        return (
            "Volatilidad elevada. "
            "Conviene incrementar la disciplina de riesgo."
        )

    return (
        "Volatilidad muy elevada. "
        "El entorno global presenta señales claras de estrés."
    )


def _interpret_dxy(
    dxy: Optional[float],
    dxy_return_20d: Optional[float],
) -> str:

    if dxy is None:
        return "Sin lectura disponible."

    if dxy_return_20d is None:
        return (
            f"DXY en {_fmt_number(dxy)}; "
            "sin tendencia reciente disponible."
        )

    if dxy_return_20d > 0.02:
        return (
            "El dólar muestra fortalecimiento reciente. "
            "Puede representar presión para activos de riesgo."
        )

    if dxy_return_20d < -0.02:
        return (
            "El dólar se está debilitando recientemente. "
            "El contexto suele ser más favorable para activos de riesgo."
        )

    return (
        "El dólar presenta un comportamiento relativamente estable."
    )


def _interpret_rsi(
    rsi: Optional[float],
) -> str:

    if rsi is None:
        return "Sin lectura RSI."

    if rsi >= 70:
        return "Sobrecompra técnica."

    if rsi <= 30:
        return "Sobreventa técnica."

    if rsi >= 55:
        return "Momentum positivo."

    if rsi <= 45:
        return "Momentum débil."

    return "Momentum neutral."


def _interpret_trend(
    close_sma20: Optional[float],
    close_sma50: Optional[float],
    close_sma200: Optional[float],
) -> str:

    values = [
        close_sma20,
        close_sma50,
        close_sma200,
    ]

    if all(
        value is None
        for value in values
    ):
        return "Tendencia no disponible."

    positive = sum(
        value is not None
        and value > 0
        for value in values
    )

    negative = sum(
        value is not None
        and value < 0
        for value in values
    )

    if positive >= 3:
        return "Tendencia estructural alcista."

    if negative >= 3:
        return "Tendencia estructural bajista."

    if positive > negative:
        return "Tendencia moderadamente alcista."

    if negative > positive:
        return "Tendencia moderadamente bajista."

    return "Tendencia mixta / lateral."


def _interpret_volatility(
    rv7: Optional[float],
    rv30: Optional[float],
) -> str:

    if rv7 is None or rv30 is None:
        return "Sin datos suficientes."

    if rv30 == 0:
        return "Sin referencia válida de volatilidad."

    ratio = rv7 / rv30

    if ratio >= 1.25:
        return (
            "Volatilidad acelerándose con fuerza "
            "en el corto plazo."
        )

    if ratio >= 1.05:
        return (
            "Volatilidad aumentando en el corto plazo."
        )

    if ratio <= 0.75:
        return (
            "Compresión clara de volatilidad "
            "en el corto plazo."
        )

    if ratio <= 0.95:
        return (
            "Volatilidad disminuyendo gradualmente."
        )

    return "Volatilidad relativamente estable."


# =============================================================================
# SCORE MACRO
# =============================================================================

def _macro_score(
    *,
    vix: Optional[float],
    dxy_return_20d: Optional[float],
    nasdaq_return_20d: Optional[float],
    benchmark_return_20d: Optional[float],
    trend_20: Optional[float],
    trend_50: Optional[float],
    rv7: Optional[float],
    rv30: Optional[float],
) -> Dict[str, Any]:

    score = 50.0
    reasons = []

    if vix is not None:
        if vix < 20:
            score += 8
            reasons.append(
                "VIX sin estrés significativo"
            )
        elif vix >= 30:
            score -= 15
            reasons.append(
                "VIX en zona de estrés"
            )
        elif vix >= 25:
            score -= 8
            reasons.append(
                "VIX elevado"
            )

    if dxy_return_20d is not None:
        if dxy_return_20d <= -0.02:
            score += 6
            reasons.append(
                "debilidad reciente del dólar"
            )
        elif dxy_return_20d >= 0.02:
            score -= 6
            reasons.append(
                "fortaleza reciente del dólar"
            )

    if nasdaq_return_20d is not None:
        if nasdaq_return_20d > 0:
            score += 6
            reasons.append(
                "Nasdaq con sesgo positivo"
            )
        elif nasdaq_return_20d < 0:
            score -= 6
            reasons.append(
                "Nasdaq con sesgo negativo"
            )

    if benchmark_return_20d is not None:
        if benchmark_return_20d >= 0.05:
            score += 8
            reasons.append(
                "benchmark con momentum fuerte"
            )
        elif benchmark_return_20d > 0:
            score += 4
            reasons.append(
                "benchmark con retorno mensual positivo"
            )
        elif benchmark_return_20d <= -0.05:
            score -= 8
            reasons.append(
                "benchmark con debilidad mensual fuerte"
            )
        else:
            score -= 4
            reasons.append(
                "benchmark con retorno mensual negativo"
            )

    trend_positive = sum(
        value is not None
        and value > 0
        for value in [
            trend_20,
            trend_50,
        ]
    )

    trend_negative = sum(
        value is not None
        and value < 0
        for value in [
            trend_20,
            trend_50,
        ]
    )

    if trend_positive == 2:
        score += 8
        reasons.append(
            "precio por encima de medias principales"
        )
    elif trend_negative == 2:
        score -= 8
        reasons.append(
            "precio por debajo de medias principales"
        )

    if (
        rv7 is not None
        and rv30 is not None
        and rv30 > 0
    ):
        ratio = rv7 / rv30

        if ratio <= 0.90:
            score += 4
            reasons.append(
                "volatilidad de corto plazo en compresión"
            )
        elif ratio >= 1.20:
            score -= 6
            reasons.append(
                "volatilidad de corto plazo acelerándose"
            )

    score = max(
        0.0,
        min(
            100.0,
            score,
        ),
    )

    if score >= 70:
        regime = "FAVORABLE"
    elif score >= 58:
        regime = "MODERADAMENTE FAVORABLE"
    elif score >= 43:
        regime = "NEUTRAL"
    elif score >= 30:
        regime = "MODERADAMENTE DESFAVORABLE"
    else:
        regime = "DESFAVORABLE"

    return {
        "score": score,
        "regime": regime,
        "reasons": reasons,
    }


# =============================================================================
# INFORME
# =============================================================================

def generar_informe_macro(
    mercado: str,
) -> str:

    manager = MarketManager(
        mercado
    )

    contexto = _load_context(
        manager
    )

    indicadores = _load_indicators(
        manager
    )

    benchmark_df = _benchmark_frame(
        manager,
        indicadores,
    )

    contexto_latest = (
        contexto.iloc[-1]
        if not contexto.empty
        else pd.Series(dtype=float)
    )

    benchmark_latest = (
        benchmark_df.iloc[-1]
        if not benchmark_df.empty
        else pd.Series(dtype=float)
    )

    # -------------------------------------------------------------------------
    # Variables macro
    # -------------------------------------------------------------------------

    vix = _latest_value(
        contexto_latest,
        "VIX",
    )

    dxy = _latest_value(
        contexto_latest,
        "DXY",
    )

    sp500 = _latest_value(
        contexto_latest,
        "SP500",
    )

    nasdaq = _latest_value(
        contexto_latest,
        "NASDAQ",
    )

    gold = _latest_value(
        contexto_latest,
        "GOLD",
    )

    oil = _latest_value(
        contexto_latest,
        "OIL_WTI",
    )

    dxy_return_20d = _latest_value(
        contexto_latest,
        "DXY_Return_20d",
    )

    nasdaq_return_20d = _latest_value(
        contexto_latest,
        "NASDAQ_Return_20d",
    )

    sp500_return_20d = _latest_value(
        contexto_latest,
        "SP500_Return_20d",
    )

    gold_return_20d = _latest_value(
        contexto_latest,
        "GOLD_Return_20d",
    )

    # -------------------------------------------------------------------------
    # Benchmark
    # -------------------------------------------------------------------------

    benchmark_close = _latest_value(
        benchmark_latest,
        "Close",
    )

    benchmark_return_1d = _latest_value(
        benchmark_latest,
        "Return_1d",
    )

    benchmark_return_5d = _latest_value(
        benchmark_latest,
        "Return_5d",
    )

    benchmark_return_20d = _latest_value(
        benchmark_latest,
        "Return_20d",
    )

    rsi = _latest_value(
        benchmark_latest,
        "RSI_14",
    )

    close_sma20 = _latest_value(
        benchmark_latest,
        "Close_vs_SMA20",
    )

    close_sma50 = _latest_value(
        benchmark_latest,
        "Close_vs_SMA50",
    )

    close_sma200 = _latest_value(
        benchmark_latest,
        "Close_vs_SMA200",
    )

    # RV calculadas por MarketManager.
    rv7 = _safe_float(
        manager.indicadores_macro.get(
            "RV_7"
        )
    )

    rv30 = _safe_float(
        manager.indicadores_macro.get(
            "RV_30"
        )
    )

    # -------------------------------------------------------------------------
    # Correlaciones reales
    # -------------------------------------------------------------------------

    corr_nasdaq_30 = (
        _rolling_correlation(
            benchmark_df,
            contexto,
            "NASDAQ",
            30,
        )
    )

    corr_nasdaq_60 = (
        _rolling_correlation(
            benchmark_df,
            contexto,
            "NASDAQ",
            60,
        )
    )

    corr_sp500_30 = (
        _rolling_correlation(
            benchmark_df,
            contexto,
            "SP500",
            30,
        )
    )

    corr_dxy_30 = (
        _rolling_correlation(
            benchmark_df,
            contexto,
            "DXY",
            30,
        )
    )

    corr_gold_30 = (
        _rolling_correlation(
            benchmark_df,
            contexto,
            "GOLD",
            30,
        )
    )

    # -------------------------------------------------------------------------
    # Interpretaciones
    # -------------------------------------------------------------------------

    trend_text = _interpret_trend(
        close_sma20,
        close_sma50,
        close_sma200,
    )

    rsi_text = _interpret_rsi(
        rsi
    )

    volatility_text = (
        _interpret_volatility(
            rv7,
            rv30,
        )
    )

    vix_text = _interpret_vix(
        vix
    )

    dxy_text = _interpret_dxy(
        dxy,
        dxy_return_20d,
    )

    macro = _macro_score(
        vix=vix,
        dxy_return_20d=dxy_return_20d,
        nasdaq_return_20d=nasdaq_return_20d,
        benchmark_return_20d=benchmark_return_20d,
        trend_20=close_sma20,
        trend_50=close_sma50,
        rv7=rv7,
        rv30=rv30,
    )

    # -------------------------------------------------------------------------
    # Fecha
    # -------------------------------------------------------------------------

    fecha_datos = None

    if (
        not benchmark_df.empty
        and "Date" in benchmark_df.columns
    ):
        fecha_datos = benchmark_df[
            "Date"
        ].max()

    if fecha_datos is None:
        fecha_texto = datetime.now().strftime(
            "%Y-%m-%d"
        )
    else:
        fecha_texto = pd.Timestamp(
            fecha_datos
        ).strftime("%Y-%m-%d")

    # -------------------------------------------------------------------------
    # Archivo salida
    # -------------------------------------------------------------------------

    output_folder = (
        Path("reportes")
        / manager.market.market
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_folder
        / "informe_macro.txt"
    )

    # -------------------------------------------------------------------------
    # Escritura
    # -------------------------------------------------------------------------

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "=" * 78
            + "\n"
        )

        f.write(
            "ATLAS / MARKET AI - INFORME MACRO Y DE REGIMEN DE MERCADO\n"
        )

        f.write(
            "=" * 78
            + "\n\n"
        )

        f.write(
            f"Fecha de datos : {fecha_texto}\n"
        )

        f.write(
            f"Mercado        : {manager.nombre}\n"
        )

        f.write(
            f"Benchmark      : {manager.benchmark}\n"
        )

        f.write(
            f"Score Macro    : {macro['score']:.1f}/100\n"
        )

        f.write(
            f"Regimen        : {macro['regime']}\n\n"
        )

        # =====================================================================
        f.write(
            "-" * 78
            + "\n"
        )

        f.write(
            "1. RESUMEN EJECUTIVO\n"
        )

        f.write(
            "-" * 78
            + "\n"
        )

        f.write(
            f"El entorno actual para {manager.nombre} se clasifica como "
            f"{macro['regime']}, con un score macro de "
            f"{macro['score']:.1f}/100.\n"
        )

        if macro["reasons"]:
            f.write(
                "Factores principales: "
                + "; ".join(
                    macro["reasons"]
                )
                + ".\n"
            )

        f.write(
            f"{vix_text}\n"
        )

        f.write(
            f"{dxy_text}\n"
        )

        f.write(
            f"{trend_text} {rsi_text}\n\n"
        )

        # =====================================================================
        f.write(
            "-" * 78
            + "\n"
        )

        f.write(
            "2. CONTEXTO GLOBAL\n"
        )

        f.write(
            "-" * 78
            + "\n"
        )

        f.write(
            f"VIX              : {_fmt_number(vix)}\n"
        )

        f.write(
            f"DXY              : {_fmt_number(dxy)}"
            f" | 20d {_fmt_pct(dxy_return_20d)}\n"
        )

        f.write(
            f"S&P 500          : {_fmt_number(sp500)}"
            f" | 20d {_fmt_pct(sp500_return_20d)}\n"
        )

        f.write(
            f"NASDAQ           : {_fmt_number(nasdaq)}"
            f" | 20d {_fmt_pct(nasdaq_return_20d)}\n"
        )

        f.write(
            f"GOLD             : {_fmt_number(gold)}"
            f" | 20d {_fmt_pct(gold_return_20d)}\n"
        )

        f.write(
            f"OIL WTI          : {_fmt_number(oil)}\n\n"
        )

        # =====================================================================
        f.write(
            "-" * 78
            + "\n"
        )

        f.write(
            "3. BENCHMARK DEL MERCADO\n"
        )

        f.write(
            "-" * 78
            + "\n"
        )

        f.write(
            f"Benchmark         : {manager.benchmark}\n"
        )

        f.write(
            f"Precio / nivel    : {_fmt_number(benchmark_close)}\n"
        )

        f.write(
            f"Retorno 1d        : {_fmt_pct(benchmark_return_1d)}\n"
        )

        f.write(
            f"Retorno 5d        : {_fmt_pct(benchmark_return_5d)}\n"
        )

        f.write(
            f"Retorno 20d       : {_fmt_pct(benchmark_return_20d)}\n"
        )

        f.write(
            f"RSI 14            : {_fmt_number(rsi)}"
            f" -> {rsi_text}\n"
        )

        f.write(
            f"Vs SMA20          : {_fmt_pct(close_sma20)}\n"
        )

        f.write(
            f"Vs SMA50          : {_fmt_pct(close_sma50)}\n"
        )

        f.write(
            f"Vs SMA200         : {_fmt_pct(close_sma200)}\n"
        )

        f.write(
            f"Lectura tendencia : {trend_text}\n\n"
        )

        # =====================================================================
        f.write(
            "-" * 78
            + "\n"
        )

        f.write(
            "4. VOLATILIDAD\n"
        )

        f.write(
            "-" * 78
            + "\n"
        )

        f.write(
            f"RV 7 días         : {_fmt_pct(rv7)}\n"
        )

        f.write(
            f"RV 30 días        : {_fmt_pct(rv30)}\n"
        )

        if (
            rv7 is not None
            and rv30 is not None
            and rv30 != 0
        ):
            f.write(
                f"Ratio RV7/RV30    : {rv7 / rv30:.2f}\n"
            )

        f.write(
            f"Lectura           : {volatility_text}\n\n"
        )

        # =====================================================================
        f.write(
            "-" * 78
            + "\n"
        )

        f.write(
            "5. CORRELACIONES DEL BENCHMARK\n"
        )

        f.write(
            "-" * 78
            + "\n"
        )

        f.write(
            f"NASDAQ 30d        : {_fmt_number(corr_nasdaq_30)}\n"
        )

        f.write(
            f"NASDAQ 60d        : {_fmt_number(corr_nasdaq_60)}\n"
        )

        f.write(
            f"S&P 500 30d       : {_fmt_number(corr_sp500_30)}\n"
        )

        f.write(
            f"DXY 30d           : {_fmt_number(corr_dxy_30)}\n"
        )

        f.write(
            f"GOLD 30d          : {_fmt_number(corr_gold_30)}\n\n"
        )

        # =====================================================================
        f.write(
            "-" * 78
            + "\n"
        )

        f.write(
            "6. MAPA DE RIESGOS\n"
        )

        f.write(
            "-" * 78
            + "\n"
        )

        risks = []

        if vix is not None and vix >= 25:
            risks.append(
                "VIX elevado"
            )

        if (
            dxy_return_20d is not None
            and dxy_return_20d >= 0.02
        ):
            risks.append(
                "fortalecimiento del dólar"
            )

        if (
            benchmark_return_20d is not None
            and benchmark_return_20d < 0
        ):
            risks.append(
                "benchmark con retorno mensual negativo"
            )

        if (
            rv7 is not None
            and rv30 is not None
            and rv30 > 0
            and rv7 / rv30 >= 1.20
        ):
            risks.append(
                "aceleración de volatilidad"
            )

        if (
            close_sma20 is not None
            and close_sma20 < 0
            and close_sma50 is not None
            and close_sma50 < 0
        ):
            risks.append(
                "debilidad técnica de corto/medio plazo"
            )

        if risks:
            for risk in risks:
                f.write(
                    f"- {risk}\n"
                )
        else:
            f.write(
                "No se detectan señales macro de estrés severo "
                "con los indicadores disponibles.\n"
            )

        f.write("\n")

        # =====================================================================
        f.write(
            "-" * 78
            + "\n"
        )

        f.write(
            "7. CONCLUSION INSTITUCIONAL\n"
        )

        f.write(
            "-" * 78
            + "\n"
        )

        f.write(
            f"Lectura general para {manager.nombre}: "
            f"{macro['regime']}.\n"
        )

        f.write(
            f"Score macro: {macro['score']:.1f}/100.\n"
        )

        f.write(
            f"El benchmark presenta: {trend_text.lower()} "
            f"{rsi_text.lower()}\n"
        )

        f.write(
            f"En volatilidad: {volatility_text.lower()}\n"
        )

        if macro["score"] >= 58:
            f.write(
                "El contexto permite asumir riesgo de forma selectiva, "
                "priorizando activos con confirmación técnica y consenso "
                "de los modelos.\n"
            )

        elif macro["score"] >= 43:
            f.write(
                "El entorno aconseja selectividad. Conviene priorizar "
                "las señales de mayor score y evitar posiciones basadas "
                "únicamente en momentum.\n"
            )

        else:
            f.write(
                "El contexto aconseja una postura defensiva, reducción "
                "de exposición y exigencia de señales fuertes antes de "
                "asumir riesgo adicional.\n"
            )

        f.write("\n")

        f.write(
            "Este informe complementa el ranking cuantitativo diario "
            "de Market AI; no sustituye las señales individuales de "
            "los modelos ni el control de riesgo.\n"
        )

    return str(
        output_file
    )


if __name__ == "__main__":
    archivo = generar_informe_macro(
        "crypto"
    )

    print(
        f"Informe generado: {archivo}"
    )
