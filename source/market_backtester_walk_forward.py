import os
import pandas as pd
import numpy as np
from datetime import datetime

from source.app_logger import log_print as print


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CAPITAL_INICIAL = 100000
HORIZONTE_DIAS = 5
TOP_ACTIVOS = 2
COSTO_TRANSACCION = 0.002


def _parse_bool_env(value, default=False):
    if value is None:
        return default

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def exportar_excel_walk_forward_habilitado():
    global_default = _parse_bool_env(os.getenv("MARKET_EXPORT_EXCEL"), default=False)
    return _parse_bool_env(os.getenv("MARKET_WALK_FORWARD_EXPORT_EXCEL"), default=global_default)


def buscar_archivo_indicadores(mercado):
    ruta = os.path.join(
        BASE_DIR,
        "datos",
        mercado.lower(),
        "procesados",
        f"{mercado.upper()}_INDICADORES.csv"
    )

    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No existe archivo de indicadores: {ruta}")

    return ruta


def cargar_datos(mercado):
    archivo = buscar_archivo_indicadores(mercado)
    df = pd.read_csv(archivo)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    return df, archivo


def detectar_columna_score(df):
    posibles = [
        "Return_20d",
        "Return_10d",
        "Return_5d",
        "Momentum_20",
        "Momentum_10",
        "Score_Tecnico",
    ]

    for col in posibles:
        if col in df.columns:
            return col

    raise ValueError("No se encontró una columna válida para score.")


def calcular_retorno_futuro(df):
    df = df.copy()
    df["Close_Futuro"] = df.groupby("Ticker")["Close"].shift(-HORIZONTE_DIAS)
    df["Close_Entrada"] = df.groupby("Ticker")["Close"].shift(-1)

    df["Retorno_Futuro"] = (
        (df["Close_Futuro"] - df["Close_Entrada"]) / df["Close_Entrada"]
    )

    return df


def backtest_walk_forward(df, score_col):
    operaciones = []
    equity = []

    capital = CAPITAL_INICIAL
    fechas = sorted(df["Date"].unique())

    for i in range(0, len(fechas), HORIZONTE_DIAS):
        fecha = fechas[i]

        dia = df[df["Date"] == fecha].copy()

        if dia.empty:
            continue

        dia = dia.dropna(subset=[score_col, "Close_Entrada", "Close_Futuro", "Retorno_Futuro"])

        if dia.empty:
            continue

        seleccion = dia.sort_values(score_col, ascending=False).head(TOP_ACTIVOS)

        if seleccion.empty:
            continue

        capital_por_activo = capital / len(seleccion)
        retorno_cartera = seleccion["Retorno_Futuro"].mean()

        costo_total = COSTO_TRANSACCION * 2
        retorno_neto = retorno_cartera - costo_total

        capital_anterior = capital
        capital = capital * (1 + retorno_neto)

        equity.append({
            "Fecha": fecha,
            "Capital": capital,
            "Retorno_Cartera": retorno_neto,
            "Capital_Anterior": capital_anterior,
            "Activos": ", ".join(seleccion["Ticker"].astype(str).tolist())
        })

        for _, row in seleccion.iterrows():
            operaciones.append({
                "Fecha_Senal": fecha,
                "Ticker": row["Ticker"],
                "Score_Usado": row[score_col],
                "Close_Senal": row["Close"],
                "Close_Entrada_T1": row["Close_Entrada"],
                "Close_Salida_T5": row["Close_Futuro"],
                "Retorno_Bruto": row["Retorno_Futuro"],
                "Retorno_Neto_Aprox": row["Retorno_Futuro"] - costo_total,
                "Capital_Asignado": capital_por_activo
            })

    return pd.DataFrame(operaciones), pd.DataFrame(equity)

def calcular_buy_and_hold(df):
    resultados = []

    for ticker, grupo in df.groupby("Ticker"):
        grupo = grupo.sort_values("Date").dropna(subset=["Close"])

        if len(grupo) < 2:
            continue

        precio_inicio = grupo.iloc[0]["Close"]
        precio_fin = grupo.iloc[-1]["Close"]

        retorno = (precio_fin - precio_inicio) / precio_inicio
        capital_final = CAPITAL_INICIAL * (1 + retorno)

        resultados.append({
            "Ticker": ticker,
            "Fecha_Inicio": grupo.iloc[0]["Date"],
            "Fecha_Fin": grupo.iloc[-1]["Date"],
            "Precio_Inicio": precio_inicio,
            "Precio_Fin": precio_fin,
            "Retorno_Total_%": retorno * 100,
            "Capital_Final": capital_final
        })

    return pd.DataFrame(resultados).sort_values("Retorno_Total_%", ascending=False)


def max_drawdown(equity):
    capital = equity["Capital"]
    peak = capital.cummax()
    dd = (capital - peak) / peak
    return dd.min() * 100


def resumen_backtest(operaciones, equity, score_col):
    if operaciones.empty or equity.empty:
        return {}

    retornos = operaciones["Retorno_Neto_Aprox"].dropna()

    ganadoras = retornos[retornos > 0]
    perdedoras = retornos[retornos <= 0]

    profit_factor = (
        ganadoras.sum() / abs(perdedoras.sum())
        if abs(perdedoras.sum()) > 0 else np.nan
    )

    capital_final = equity.iloc[-1]["Capital"]
    retorno_total = (capital_final / CAPITAL_INICIAL - 1) * 100

    resumen = {
        "capital_inicial": CAPITAL_INICIAL,
        "capital_final": round(capital_final, 2),
        "retorno_total_%": round(retorno_total, 2),
        "operaciones": len(operaciones),
        "dias_evaluados": len(equity),
        "top_activos_por_fecha": TOP_ACTIVOS,
        "horizonte_dias": HORIZONTE_DIAS,
        "costo_transaccion": COSTO_TRANSACCION,
        "win_rate_%": round((retornos > 0).mean() * 100, 2),
        "retorno_promedio_%": round(retornos.mean() * 100, 2),
        "ganancia_promedio_%": round(ganadoras.mean() * 100, 2),
        "perdida_promedio_%": round(perdedoras.mean() * 100, 2),
        "profit_factor": round(profit_factor, 3),
        "max_drawdown_%": round(max_drawdown(equity), 2),
        "score_usado": score_col,
        "metodo": "Walk Forward sin look-ahead: señal en t, entrada en t+1, salida en t+5"
    }

    return resumen


def guardar_reportes(mercado, operaciones, equity, buy_hold, resumen):
    carpeta = os.path.join(BASE_DIR, "reportes", mercado.lower())
    os.makedirs(carpeta, exist_ok=True)

    path_ops = os.path.join(carpeta, f"walk_forward_operaciones_{mercado.lower()}.csv")
    path_equity = os.path.join(carpeta, f"walk_forward_equity_{mercado.lower()}.csv")
    path_bh = os.path.join(carpeta, f"walk_forward_buy_hold_{mercado.lower()}.csv")
    path_resumen = os.path.join(carpeta, f"walk_forward_resumen_{mercado.lower()}.csv")
    path_txt = os.path.join(carpeta, f"walk_forward_resumen_{mercado.lower()}.txt")
    path_xlsx = os.path.join(carpeta, f"walk_forward_backtest_{mercado.lower()}.xlsx")

    operaciones.to_csv(path_ops, index=False)
    equity.to_csv(path_equity, index=False)
    buy_hold.to_csv(path_bh, index=False)
    pd.DataFrame([resumen]).to_csv(path_resumen, index=False)

    excel_resumen = "OMITIDO (usar --excel-all para habilitar)"
    if exportar_excel_walk_forward_habilitado():
        with pd.ExcelWriter(path_xlsx, engine="openpyxl") as writer:
            pd.DataFrame([resumen]).to_excel(writer, sheet_name="Resumen", index=False)
            equity.to_excel(writer, sheet_name="Equity", index=False)
            operaciones.to_excel(writer, sheet_name="Operaciones", index=False)
            buy_hold.to_excel(writer, sheet_name="BuyHold", index=False)
        excel_resumen = path_xlsx

    with open(path_txt, "w", encoding="utf-8") as f:
        f.write("MARKET AI WALK FORWARD BACKTEST\n")
        f.write("=" * 80 + "\n\n")

        for k, v in resumen.items():
            f.write(f"{k}: {v}\n")

        f.write("\nTOP BUY & HOLD\n")
        f.write("-" * 80 + "\n")
        f.write(buy_hold.head(10).to_string(index=False))

    return {
        "operaciones": path_ops,
        "equity": path_equity,
        "buy_hold": path_bh,
        "resumen": path_resumen,
        "txt": path_txt,
        "excel": excel_resumen
    }


def ejecutar_backtest(mercado):
    print("=" * 80)
    print(f"MARKET AI WALK FORWARD BACKTEST - {mercado.upper()}")
    print("=" * 80)

    df, archivo = cargar_datos(mercado)

    print(f"Archivo indicadores : {archivo}")
    print(f"Registros           : {len(df):,}")
    print(f"Activos             : {df['Ticker'].nunique()}")
    print(f"Desde               : {df['Date'].min()}")
    print(f"Hasta               : {df['Date'].max()}")

    score_col = detectar_columna_score(df)

    print(f"Score usado         : {score_col}")
    print(f"Horizonte           : {HORIZONTE_DIAS} días")
    print(f"Top activos         : {TOP_ACTIVOS}")
    print(f"Costo transacción   : {COSTO_TRANSACCION * 100:.2f}%")

    df = calcular_retorno_futuro(df)

    operaciones, equity = backtest_walk_forward(df, score_col)
    buy_hold = calcular_buy_and_hold(df)
    resumen = resumen_backtest(operaciones, equity, score_col)

    archivos = guardar_reportes(mercado, operaciones, equity, buy_hold, resumen)

    print("\n" + "=" * 80)
    print("RESULTADOS WALK FORWARD")
    print("=" * 80)

    for k, v in resumen.items():
        print(f"{k:25}: {v}")

    print("\n" + "=" * 80)
    print("TOP BUY & HOLD")
    print("=" * 80)
    print(buy_hold.head(10).to_string(index=False))

    print("\n" + "=" * 80)
    print("ARCHIVOS GENERADOS")
    print("=" * 80)

    for path in archivos.values():
        print(path)

    print("\n✅ WALK FORWARD BACKTEST FINALIZADO CORRECTAMENTE")


if __name__ == "__main__":
    import sys

    mercados = {
        "1": "merval",
        "2": "crypto",
        "3": "nasdaq",
        "4": "etf",
        "5": "forex",
        "6": "commodities",
        "7": "bonos",
        "8": "indices",
    }

    if len(sys.argv) > 1:
        mercado_arg = sys.argv[1].lower()

        if mercado_arg in mercados.values():
            ejecutar_backtest(mercado_arg)
        else:
            print(f"Mercado inválido: {mercado_arg}")

        sys.exit()

    print("=" * 80)
    print("SELECCIONE MERCADO PARA WALK FORWARD BACKTEST")
    print("=" * 80)
    print("1. MERVAL")
    print("2. CRYPTO")
    print("3. NASDAQ")
    print("4. ETF")
    print("5. FOREX")
    print("6. COMMODITIES")
    print("7. BONOS")
    print("8. INDICES")
    print("0. Salir")
    print("=" * 80)

    opcion = input("Ingrese opción: ").strip()

    if opcion == "0":
        print("Salida.")
    elif opcion in mercados:
        ejecutar_backtest(mercados[opcion])
    else:
        print("Opción inválida.")