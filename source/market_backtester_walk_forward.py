# source/market_backtester_walk_forward.py
"""Simulación forward sobre predicciones IA estrictamente fuera de muestra.

Consume ``backtest_ai_scores_<mercado>.csv``, generado por
``market_backtester.py`` mediante una selección anidada. No usa Return_20d ni
recalcula pesos con información del período evaluado.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from config import MarketConfig


CAPITAL_INICIAL = 100000.0
HORIZONTE_DIAS = 5
TOP_ACTIVOS = 2
COSTO_POR_LADO = 0.002
COSTO_IDA_VUELTA = COSTO_POR_LADO * 2
SENALES_OPERABLES = {"COMPRA", "COMPRA FUERTE"}


def rutas_entrada(market):
    reportes = ROOT / "reportes" / market.market
    scores = reportes / f"backtest_ai_scores_{market.market}.csv"
    resumen = reportes / f"backtest_resumen_{market.market}.csv"
    return scores, resumen


def cargar_scores_oos(market):
    scores_path, resumen_path = rutas_entrada(market)
    if not scores_path.exists() or not resumen_path.exists():
        raise FileNotFoundError(
            "No existen predicciones IA OOS. Ejecutá primero "
            f"market_backtester.py {market.market}"
        )

    df = pd.read_csv(scores_path)
    metadata = pd.read_csv(resumen_path)
    obligatorias = {
        "Date", "Ticker", "Close", "Score_Final", "Consensus_Probability",
        "Probability_Edge", "Señal", "Close_Entrada_T1", "Close_Salida_T5",
        "Retorno_Futuro",
    }
    faltantes = sorted(obligatorias - set(df.columns))
    if faltantes:
        raise ValueError(
            f"El archivo OOS no fue generado por el backtester IA nuevo. Faltan: {faltantes}"
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in [
        "Close", "Score_Final", "Consensus_Probability", "Probability_Edge",
        "Close_Entrada_T1", "Close_Salida_T5", "Retorno_Futuro",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Date", "Ticker", "Score_Final"]).copy()
    df.sort_values(["Date", "Ticker"], inplace=True)

    if metadata.empty:
        raise ValueError("El resumen del backtest IA está vacío.")
    meta = metadata.iloc[0].to_dict()
    if str(meta.get("tipo_backtest", "")) != "ENSEMBLE_IA_ANIDADO_OOS":
        raise ValueError("El archivo de scores no posee validación anidada OOS.")
    return df, meta, scores_path


def simular_walk_forward(df):
    operaciones = []
    equity = []
    capital = CAPITAL_INICIAL
    fechas = sorted(df["Date"].dropna().unique())[::HORIZONTE_DIAS]

    for fecha in fechas:
        dia = df[df["Date"] == fecha].dropna(
            subset=["Score_Final", "Close_Entrada_T1", "Close_Salida_T5", "Retorno_Futuro"]
        ).copy()
        if dia.empty:
            continue

        seleccion = (
            dia[dia["Señal"].isin(SENALES_OPERABLES)]
            .sort_values("Score_Final", ascending=False)
            .head(TOP_ACTIVOS)
        )
        if seleccion.empty:
            equity.append({
                "Fecha": fecha,
                "Capital": capital,
                "Retorno_Cartera": 0.0,
                "Capital_Anterior": capital,
                "Activos": "CASH",
                "Score_Usado": "Score_Final_IA_OOS",
                "Estado": "CASH_SIN_SEÑAL",
            })
            continue

        capital_por_activo = capital / len(seleccion)
        retorno_bruto = float(seleccion["Retorno_Futuro"].mean())
        retorno_neto = retorno_bruto - COSTO_IDA_VUELTA
        anterior = capital
        capital *= 1.0 + retorno_neto

        equity.append({
            "Fecha": fecha,
            "Capital": capital,
            "Retorno_Cartera": retorno_neto,
            "Capital_Anterior": anterior,
            "Activos": ", ".join(seleccion["Ticker"].astype(str)),
            "Score_Usado": "Score_Final_IA_OOS",
            "Estado": "INVERTIDO",
        })

        for _, row in seleccion.iterrows():
            operaciones.append({
                "Fecha_Senal": fecha,
                "Ticker": row["Ticker"],
                "Score_Final_IA": row["Score_Final"],
                "Consensus_Probability": row["Consensus_Probability"],
                "Probability_Edge": row["Probability_Edge"],
                "Señal": row["Señal"],
                "Close_Senal": row["Close"],
                "Close_Entrada_T1": row["Close_Entrada_T1"],
                "Close_Salida_T5": row["Close_Salida_T5"],
                "Retorno_Bruto": row["Retorno_Futuro"],
                "Retorno_Neto_Aprox": row["Retorno_Futuro"] - COSTO_IDA_VUELTA,
                "Capital_Asignado": capital_por_activo,
            })

    return pd.DataFrame(operaciones), pd.DataFrame(equity)


def calcular_buy_and_hold(market, fecha_inicio, fecha_fin):
    df = pd.read_csv(market.dataset_csv, usecols=["Date", "Ticker", "Close"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df[df["Date"].between(pd.Timestamp(fecha_inicio), pd.Timestamp(fecha_fin))]
    resultados = []
    for ticker, grupo in df.groupby("Ticker"):
        grupo = grupo.sort_values("Date").dropna(subset=["Date", "Close"])
        if len(grupo) < 2:
            continue
        inicio = grupo.iloc[0]
        fin = grupo.iloc[-1]
        retorno = float(fin["Close"] / inicio["Close"] - 1.0)
        resultados.append({
            "Ticker": ticker,
            "Fecha_Inicio": inicio["Date"],
            "Fecha_Fin": fin["Date"],
            "Precio_Inicio": inicio["Close"],
            "Precio_Fin": fin["Close"],
            "Retorno_Total_%": retorno * 100.0,
            "Capital_Final": CAPITAL_INICIAL * (1.0 + retorno),
        })
    return pd.DataFrame(resultados).sort_values("Retorno_Total_%", ascending=False)


def max_drawdown(equity):
    capital = pd.concat(
        [pd.Series([CAPITAL_INICIAL], dtype=float), equity["Capital"].astype(float)],
        ignore_index=True,
    )
    peak = capital.cummax()
    return float(((capital - peak) / peak).min() * 100.0)


def resumen_backtest(operaciones, equity, metadata):
    if equity.empty:
        raise RuntimeError("La simulación OOS no produjo períodos evaluables.")

    retornos_ops = (
        operaciones["Retorno_Neto_Aprox"].dropna()
        if not operaciones.empty else pd.Series(dtype=float)
    )
    retornos_cartera = equity.loc[
        equity["Estado"] == "INVERTIDO", "Retorno_Cartera"
    ].dropna()
    ganadoras = retornos_ops[retornos_ops > 0]
    perdedoras = retornos_ops[retornos_ops <= 0]
    profit_factor = (
        ganadoras.sum() / abs(perdedoras.sum())
        if len(retornos_ops) and abs(perdedoras.sum()) > 0 else np.nan
    )
    capital_final = float(equity.iloc[-1]["Capital"])
    dias_totales = max(
        (pd.Timestamp(equity["Fecha"].max()) - pd.Timestamp(equity["Fecha"].min())).days,
        1,
    )
    years = dias_totales / 365.25
    cagr = (capital_final / CAPITAL_INICIAL) ** (1.0 / years) - 1.0

    return {
        "tipo_backtest": "WALK_FORWARD_SOBRE_PREDICCIONES_IA_OOS",
        "capital_inicial": CAPITAL_INICIAL,
        "capital_final": round(capital_final, 2),
        "retorno_total_%": round((capital_final / CAPITAL_INICIAL - 1.0) * 100.0, 2),
        "cagr_%": round(cagr * 100.0, 2),
        "operaciones": len(operaciones),
        "periodos_evaluados": len(equity),
        "periodos_con_posicion": int((equity["Estado"] == "INVERTIDO").sum()),
        "periodos_cash": int((equity["Estado"] == "CASH_SIN_SEÑAL").sum()),
        "participacion_mercado_pct": round(
            (equity["Estado"] == "INVERTIDO").mean() * 100.0, 2
        ),
        "top_activos_por_fecha": TOP_ACTIVOS,
        "horizonte_dias": HORIZONTE_DIAS,
        "costo_por_lado": COSTO_POR_LADO,
        "costo_ida_vuelta": COSTO_IDA_VUELTA,
        "win_rate_operaciones_%": round((retornos_ops > 0).mean() * 100.0, 2) if len(retornos_ops) else 0.0,
        "win_rate_carteras_%": round((retornos_cartera > 0).mean() * 100.0, 2) if len(retornos_cartera) else 0.0,
        "retorno_promedio_operacion_%": round(retornos_ops.mean() * 100.0, 2) if len(retornos_ops) else 0.0,
        "retorno_promedio_cartera_%": round(retornos_cartera.mean() * 100.0, 2) if len(retornos_cartera) else 0.0,
        "ganancia_promedio_%": round(ganadoras.mean() * 100.0, 2) if len(ganadoras) else 0.0,
        "perdida_promedio_%": round(perdedoras.mean() * 100.0, 2) if len(perdedoras) else 0.0,
        "profit_factor": round(float(profit_factor), 3) if pd.notna(profit_factor) else None,
        "max_drawdown_%": round(max_drawdown(equity), 2),
        "score_usado": "Score_Final_IA_OOS",
        "politica_operativa": "SOLO_COMPRA_Y_COMPRA_FUERTE; CASH_EN_OTRO_CASO",
        "modelos_usados": metadata.get("modelos_usados"),
        "modo_seleccion": metadata.get("modo_seleccion"),
        "test_oos_desde": metadata.get("test_oos_desde"),
        "test_oos_hasta": metadata.get("test_oos_hasta"),
        "separacion_temporal_estricta": True,
        "metodo": "señal IA en t, entrada en t+1, salida en t+5; pesos fijados antes del test",
    }


def guardar_reportes(market, operaciones, equity, buy_hold, resumen):
    carpeta = ROOT / "reportes" / market.market
    carpeta.mkdir(parents=True, exist_ok=True)
    paths = {
        "operaciones": carpeta / f"walk_forward_operaciones_{market.market}.csv",
        "equity": carpeta / f"walk_forward_equity_{market.market}.csv",
        "buy_hold": carpeta / f"walk_forward_buy_hold_{market.market}.csv",
        "resumen": carpeta / f"walk_forward_resumen_{market.market}.csv",
        "txt": carpeta / f"walk_forward_resumen_{market.market}.txt",
        "excel": carpeta / f"walk_forward_backtest_{market.market}.xlsx",
    }
    operaciones.to_csv(paths["operaciones"], index=False, encoding="utf-8-sig")
    equity.to_csv(paths["equity"], index=False, encoding="utf-8-sig")
    buy_hold.to_csv(paths["buy_hold"], index=False, encoding="utf-8-sig")
    pd.DataFrame([resumen]).to_csv(paths["resumen"], index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(paths["excel"], engine="openpyxl") as writer:
        pd.DataFrame([resumen]).to_excel(writer, sheet_name="Resumen", index=False)
        equity.to_excel(writer, sheet_name="Equity", index=False)
        operaciones.to_excel(writer, sheet_name="Operaciones", index=False)
        buy_hold.to_excel(writer, sheet_name="BuyHold", index=False)
    with open(paths["txt"], "w", encoding="utf-8") as f:
        f.write(f"MARKET AI WALK FORWARD OOS - {market.display_name}\n")
        f.write("=" * 90 + "\n\n")
        for key, value in resumen.items():
            f.write(f"{key:34}: {value}\n")
        f.write("\nTOP BUY & HOLD\n")
        f.write("-" * 90 + "\n")
        f.write(buy_hold.head(10).to_string(index=False))
    return paths


def ejecutar_backtest(mercado="merval"):
    inicio = datetime.now()
    market = MarketConfig(mercado)
    print("=" * 90)
    print(f"MARKET AI WALK FORWARD OOS - {market.display_name}")
    print("=" * 90)

    df, metadata, scores_path = cargar_scores_oos(market)
    print(f"Predicciones IA OOS    : {scores_path}")
    print(f"Registros OOS          : {len(df):,}")
    print(f"Activos                : {df['Ticker'].nunique()}")
    print(f"Desde                  : {df['Date'].min()}")
    print(f"Hasta                  : {df['Date'].max()}")
    print("Score usado            : Score_Final_IA_OOS")
    print(f"Top activos            : {TOP_ACTIVOS}")
    print(f"Costo ida y vuelta     : {COSTO_IDA_VUELTA * 100:.2f}%")

    operaciones, equity = simular_walk_forward(df)
    buy_hold = calcular_buy_and_hold(
        market,
        fecha_inicio=df["Date"].min(),
        fecha_fin=df["Date"].max(),
    )
    resumen = resumen_backtest(operaciones, equity, metadata)
    paths = guardar_reportes(market, operaciones, equity, buy_hold, resumen)

    print()
    print("=" * 90)
    print("RESULTADOS WALK FORWARD IA OOS")
    print("=" * 90)
    for key, value in resumen.items():
        print(f"{key:34}: {value}")
    print()
    print("TOP BUY & HOLD")
    print(buy_hold.head(10).to_string(index=False))
    print()
    print("ARCHIVOS GENERADOS")
    for path in paths.values():
        print(path)
    print(f"Tiempo total                           : {datetime.now() - inicio}")
    print("OK WALK FORWARD IA OOS FINALIZADO CORRECTAMENTE")
    return resumen, operaciones, equity


if __name__ == "__main__":
    mercado_arg = sys.argv[1].lower() if len(sys.argv) > 1 else "merval"
    ejecutar_backtest(mercado_arg)
