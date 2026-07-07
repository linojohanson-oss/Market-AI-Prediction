# source/market_backtester.py

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from config import MarketConfig


CAPITAL_INICIAL = 100000.0
TOP_N = 2
HORIZONTE_DIAS = 5
COSTO_TRANSACCION = 0.002


def resolver_columna(df, posibles):
    for col in posibles:
        if col in df.columns:
            return col
    return None


def normalizar_serie(s):
    s = pd.to_numeric(s, errors="coerce").fillna(0)

    minimo = s.min()
    maximo = s.max()

    if maximo - minimo == 0:
        return pd.Series(0.5, index=s.index)

    return (s - minimo) / (maximo - minimo)


def cargar_indicadores(mercado):
    market = MarketConfig(mercado)
    path = (
        ROOT
        / "datos"
        / market.market
        / "procesados"
        / f"{market.market.upper()}_INDICADORES.csv"
    )

    if not path.exists():
        raise FileNotFoundError(f"No existe archivo de indicadores: {path}")

    df = pd.read_csv(path)
    return df, path, market


def preparar_datos(df):
    col_ticker = resolver_columna(df, ["Ticker", "ticker", "Symbol", "symbol"])
    col_fecha = resolver_columna(df, ["Fecha", "fecha", "Date", "date"])
    col_close = resolver_columna(df, ["Close", "close", "Adj Close", "adj_close"])

    if col_ticker is None:
        raise ValueError("No encontré columna de ticker.")

    if col_fecha is None:
        raise ValueError("No encontré columna de fecha.")

    if col_close is None:
        raise ValueError("No encontré columna de precio Close.")

    df = df.copy()

    df[col_fecha] = pd.to_datetime(df[col_fecha], errors="coerce")
    df[col_close] = pd.to_numeric(df[col_close], errors="coerce")

    df = df.dropna(subset=[col_ticker, col_fecha, col_close])
    df = df.sort_values([col_ticker, col_fecha]).reset_index(drop=True)

    df["Return_5d_calc"] = (
        df.groupby(col_ticker)[col_close]
        .shift(-HORIZONTE_DIAS)
        .div(df[col_close])
        .sub(1)
    )

    return df, col_ticker, col_fecha, col_close


def detectar_score(df):
    posibles = [
        "Score_Final",
        "score_final",
        "Score_Tecnico",
        "score_tecnico",
        "Consensus_Score",
        "consensus_score",
        "Momentum_20",
        "momentum_20",
        "Return_20d",
        "return_20d",
        "RSI_14",
        "rsi_14",
        "MACD",
        "macd",
    ]

    col_score = resolver_columna(df, posibles)

    if col_score is not None:
        return col_score

    candidatos = [
        c for c in df.columns
        if any(
            x in c.lower()
            for x in ["score", "momentum", "return", "rsi", "macd"]
        )
    ]

    if candidatos:
        return candidatos[0]

    raise ValueError(
        "No encontré columna útil para score técnico. "
        f"Columnas disponibles: {df.columns.tolist()}"
    )


def generar_fechas_rebalanceo(df, col_fecha):
    fechas = sorted(df[col_fecha].dropna().unique())
    return fechas[::HORIZONTE_DIAS]


def construir_cartera_fecha(df, fecha, col_ticker, col_fecha, col_score):
    grupo = df[df[col_fecha] == fecha].copy()
    grupo = grupo.dropna(subset=["Return_5d_calc"])

    if len(grupo) < TOP_N:
        return None, None

    grupo[col_score] = pd.to_numeric(grupo[col_score], errors="coerce").fillna(0)
    grupo = grupo.sort_values(col_score, ascending=False).head(TOP_N)

    retorno_bruto = grupo["Return_5d_calc"].mean()
    retorno_neto = retorno_bruto - COSTO_TRANSACCION

    carteras_row = {
        "Fecha": fecha,
        "Tickers": " | ".join(grupo[col_ticker].astype(str).tolist()),
        "Score_Usado": col_score,
        "Scores": " | ".join(f"{x:.4f}" for x in grupo[col_score].tolist()),
        "Cantidad_Activos": len(grupo),
        "Retorno_Bruto_5d": retorno_bruto,
        "Costo_Transaccion": COSTO_TRANSACCION,
        "Retorno_Neto_5d": retorno_neto,
    }

    operaciones = []

    for _, row in grupo.iterrows():
        operaciones.append({
            "Fecha": fecha,
            "Ticker": row[col_ticker],
            "Score_Usado": col_score,
            "Score": row[col_score],
            "Return_5d": row["Return_5d_calc"],
        })

    return carteras_row, operaciones


def generar_backtest(df, col_ticker, col_fecha):
    col_score = detectar_score(df)
    fechas_rebalanceo = generar_fechas_rebalanceo(df, col_fecha)

    carteras = []
    operaciones = []

    for fecha in fechas_rebalanceo:
        cartera_row, ops = construir_cartera_fecha(
            df=df,
            fecha=fecha,
            col_ticker=col_ticker,
            col_fecha=col_fecha,
            col_score=col_score,
        )

        if cartera_row is None:
            continue

        carteras.append(cartera_row)
        operaciones.extend(ops)

    carteras_df = pd.DataFrame(carteras)
    operaciones_df = pd.DataFrame(operaciones)

    return carteras_df, operaciones_df, col_score


def calcular_equity(carteras):
    capital = CAPITAL_INICIAL
    rows = []

    for _, row in carteras.iterrows():
        ret = float(row["Retorno_Neto_5d"])

        capital_anterior = capital
        capital = capital * (1 + ret)

        rows.append({
            "Fecha": row["Fecha"],
            "Capital_Inicial_Periodo": capital_anterior,
            "Retorno_Neto_5d": ret,
            "Capital_Final_Periodo": capital,
        })

    return pd.DataFrame(rows)


def calcular_metricas(carteras, equity):
    if carteras.empty or equity.empty:
        return {}

    r = carteras["Retorno_Neto_5d"].dropna()

    ganadoras = r[r > 0]
    perdedoras = r[r <= 0]

    capital_final = equity["Capital_Final_Periodo"].iloc[-1]

    running_max = equity["Capital_Final_Periodo"].cummax()
    drawdown = (equity["Capital_Final_Periodo"] / running_max) - 1

    dias_totales = (
        pd.to_datetime(carteras["Fecha"].max())
        - pd.to_datetime(carteras["Fecha"].min())
    ).days

    years = max(dias_totales / 365.25, 1 / 365.25)

    cagr = (capital_final / CAPITAL_INICIAL) ** (1 / years) - 1

    profit_factor = (
        ganadoras.sum() / abs(perdedoras.sum())
        if abs(perdedoras.sum()) > 0 else np.nan
    )

    sharpe = (
        r.mean() / r.std() * np.sqrt(252 / HORIZONTE_DIAS)
        if r.std() > 0 else np.nan
    )

    downside = r[r < 0].std()
    sortino = (
        r.mean() / downside * np.sqrt(252 / HORIZONTE_DIAS)
        if downside and downside > 0 else np.nan
    )

    return {
        "carteras_evaluadas": len(r),
        "top_activos_por_fecha": TOP_N,
        "horizonte_dias": HORIZONTE_DIAS,
        "costo_transaccion": COSTO_TRANSACCION,
        "ganadoras": int((r > 0).sum()),
        "perdedoras": int((r <= 0).sum()),
        "win_rate": round((r > 0).mean() * 100, 2),
        "retorno_promedio_5d": round(r.mean() * 100, 2),
        "ganancia_promedio": round(ganadoras.mean() * 100, 2) if len(ganadoras) else 0,
        "perdida_promedio": round(perdedoras.mean() * 100, 2) if len(perdedoras) else 0,
        "profit_factor": round(float(profit_factor), 3) if not np.isnan(profit_factor) else None,
        "sharpe_aprox": round(float(sharpe), 3) if not np.isnan(sharpe) else None,
        "sortino_aprox": round(float(sortino), 3) if not np.isnan(sortino) else None,
        "capital_inicial": CAPITAL_INICIAL,
        "capital_final": round(float(capital_final), 2),
        "retorno_total": round((capital_final / CAPITAL_INICIAL - 1) * 100, 2),
        "cagr": round(cagr * 100, 2),
        "max_drawdown": round(drawdown.min() * 100, 2),
    }


def exportar_resultados(market, metricas, carteras, operaciones, equity, inicio):
    reportes_dir = ROOT / "reportes" / market.market
    reportes_dir.mkdir(parents=True, exist_ok=True)

    out_carteras = reportes_dir / f"backtest_carteras_{market.market}.csv"
    out_ops = reportes_dir / f"backtest_operaciones_{market.market}.csv"
    out_equity = reportes_dir / f"backtest_equity_{market.market}.csv"
    out_res = reportes_dir / f"backtest_resumen_{market.market}.csv"
    out_xlsx = reportes_dir / f"backtest_market_ai_{market.market}.xlsx"
    out_txt = reportes_dir / f"backtest_market_ai_{market.market}.txt"

    resumen_df = pd.DataFrame([metricas])

    carteras.to_csv(out_carteras, index=False, encoding="utf-8-sig")
    operaciones.to_csv(out_ops, index=False, encoding="utf-8-sig")
    equity.to_csv(out_equity, index=False, encoding="utf-8-sig")
    resumen_df.to_csv(out_res, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        resumen_df.to_excel(writer, index=False, sheet_name="Resumen")
        equity.to_excel(writer, index=False, sheet_name="Equity")
        carteras.to_excel(writer, index=False, sheet_name="Carteras")
        operaciones.to_excel(writer, index=False, sheet_name="Operaciones")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"MARKET AI BACKTESTER - {market.display_name}\n")
        f.write("=" * 80 + "\n\n")

        for k, v in metricas.items():
            f.write(f"{k:25}: {v}\n")

        f.write("\n")
        f.write(f"Archivo carteras    : {out_carteras}\n")
        f.write(f"Archivo operaciones : {out_ops}\n")
        f.write(f"Archivo equity      : {out_equity}\n")
        f.write(f"Archivo resumen     : {out_res}\n")
        f.write(f"Archivo Excel       : {out_xlsx}\n")
        f.write(f"Tiempo total        : {datetime.now() - inicio}\n")

    return {
        "carteras": out_carteras,
        "operaciones": out_ops,
        "equity": out_equity,
        "resumen": out_res,
        "xlsx": out_xlsx,
        "txt": out_txt,
    }


def ejecutar_backtest(mercado="merval"):
    inicio = datetime.now()

    print("=" * 80)
    print(f"MARKET AI BACKTESTER - {mercado.upper()}")
    print("=" * 80)

    df, path, market = cargar_indicadores(mercado)

    print(f"Archivo indicadores: {path}")
    print(f"Registros          : {len(df):,}")
    print(f"Capital inicial    : ${CAPITAL_INICIAL:,.2f}")
    print(f"Top activos        : {TOP_N}")
    print(f"Horizonte          : {HORIZONTE_DIAS} días")
    print(f"Costo transacción  : {COSTO_TRANSACCION * 100:.2f}%")

    df, col_ticker, col_fecha, col_close = preparar_datos(df)
    carteras, operaciones, col_score = generar_backtest(df, col_ticker, col_fecha)
    equity = calcular_equity(carteras)
    metricas = calcular_metricas(carteras, equity)

    metricas["score_usado"] = col_score

    archivos = exportar_resultados(
        market=market,
        metricas=metricas,
        carteras=carteras,
        operaciones=operaciones,
        equity=equity,
        inicio=inicio,
    )

    print()
    print("=" * 80)
    print("RESULTADOS BACKTEST")
    print("=" * 80)

    for k, v in metricas.items():
        print(f"{k:25}: {v}")

    print()
    print("=" * 80)
    print("ARCHIVOS GENERADOS")
    print("=" * 80)

    for _, path in archivos.items():
        print(path)

    print()
    print("=" * 80)
    print("BACKTEST FINALIZADO")
    print("=" * 80)
    print(f"Tiempo total: {datetime.now() - inicio}")

    return metricas, carteras, operaciones, equity


if __name__ == "__main__":
    ejecutar_backtest("merval")