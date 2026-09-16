# source/market_backtester.py
"""Backtest probabilístico anidado y fuera de muestra para Market AI.

La selección de modelos y pesos se realiza en una validación interna. El
período exterior de prueba permanece intacto hasta la evaluación final, por lo
que sus retornos no intervienen en el entrenamiento ni en la ponderación.
"""

import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from config import MarketConfig
from source.market_model_manager import (
    AUC_MINIMO,
    BALANCED_ACCURACY_MINIMA,
    ECE_MAXIMO,
)
from source.market_scoring import calcular_score_tecnico, clasificar_senal
from source.market_training import (
    construir_modelos,
    crear_split_temporal,
    ejecutar_con_progreso,
    expected_calibration_error,
    limpiar_X,
    probabilidades_modelo,
)


CAPITAL_INICIAL = 100000.0
TOP_N = 2
TOP_MODELOS = 3
HORIZONTE_DIAS = 5
COSTO_POR_LADO = 0.002
COSTO_IDA_VUELTA = COSTO_POR_LADO * 2
OBJETIVO = "Objetivo_5d"
TEST_EXTERNO = 0.25
VALIDACION_INTERNA = 0.25
PURGA_RUEDAS = 5
SENALES_OPERABLES = {"COMPRA", "COMPRA FUERTE"}


def cargar_dataset(market):
    if not market.dataset_csv.exists():
        raise FileNotFoundError(f"No existe dataset IA: {market.dataset_csv}")

    df = pd.read_csv(market.dataset_csv)
    obligatorias = {"Date", "Ticker", "Close", OBJETIVO}
    faltantes = sorted(obligatorias - set(df.columns))
    if faltantes:
        raise ValueError(f"Faltan columnas obligatorias en dataset IA: {faltantes}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Ticker", "Close", OBJETIVO]).copy()
    df.sort_values(["Date", "Ticker"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def columnas_features(df):
    excluir = {
        "Date", "Ticker", "Nombre", "Mercado", "Moneda", "Fuente", OBJETIVO,
    }
    return [c for c in df.columns if c not in excluir]


def calcular_metricas_modelo(nombre, y_true, pred, prob):
    tasa_real = float(np.mean(y_true))
    tasa_pred = float(np.mean(pred))
    accuracy = accuracy_score(y_true, pred)
    precision = precision_score(y_true, pred, zero_division=0)
    recall = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)
    balanced = balanced_accuracy_score(y_true, pred)
    mcc = matthews_corrcoef(y_true, pred)
    brier = brier_score_loss(y_true, prob)
    ece = expected_calibration_error(y_true, prob)
    try:
        auc = roc_auc_score(y_true, prob)
    except ValueError:
        auc = 0.5

    brier_ref = max(tasa_real * (1.0 - tasa_real), 1e-9)
    auc_skill = np.clip((auc - 0.50) / 0.50, 0.0, 1.0)
    balanced_skill = np.clip((balanced - 0.50) / 0.50, 0.0, 1.0)
    mcc_skill = np.clip(mcc, 0.0, 1.0)
    brier_skill = np.clip(1.0 - brier / brier_ref, 0.0, 1.0)
    factor_calibracion = np.clip(1.0 - ece / 0.25, 0.25, 1.0)

    if tasa_pred < 0.20:
        penalizacion = max(0.25, tasa_pred / 0.20)
    elif tasa_pred > 0.80:
        penalizacion = max(0.25, (1.0 - tasa_pred) / 0.20)
    else:
        penalizacion = 1.0

    score_base = (
        auc_skill * 0.40
        + balanced_skill * 0.25
        + mcc_skill * 0.20
        + brier_skill * 0.15
    )
    score = score_base * factor_calibracion * penalizacion

    elegible = (
        auc >= AUC_MINIMO
        and balanced >= BALANCED_ACCURACY_MINIMA
        and mcc > 0.0
        and ece <= ECE_MAXIMO
        and 0.20 <= tasa_pred <= 0.80
        and brier <= brier_ref * 1.10
    )

    razones = []
    if auc < AUC_MINIMO:
        razones.append(f"AUC<{AUC_MINIMO:.3f}")
    if balanced < BALANCED_ACCURACY_MINIMA:
        razones.append("balanced_accuracy<0.50")
    if mcc <= 0.0:
        razones.append("MCC<=0")
    if ece > ECE_MAXIMO:
        razones.append(f"ECE>{ECE_MAXIMO:.2f}")
    if not 0.20 <= tasa_pred <= 0.80:
        razones.append("colapso de clase")
    if brier > brier_ref * 1.10:
        razones.append("Brier excesivo")

    return {
        "modelo": nombre,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced,
        "mcc": mcc,
        "positive_rate_true": tasa_real,
        "positive_rate_pred": tasa_pred,
        "brier_score": brier,
        "roc_auc": auc,
        "ece_10": ece,
        "score_modelo": score,
        "elegible": elegible,
        "motivo": "ELEGIBLE" if elegible else "; ".join(razones),
    }


def seleccionar_modelos(inner_train, inner_val, features):
    X_train = limpiar_X(inner_train, features)
    y_train = inner_train[OBJETIVO].astype(int)
    X_val = limpiar_X(inner_val, features)
    y_val = inner_val[OBJETIVO].astype(int)

    resultados = []
    plantillas = construir_modelos()
    total = len(plantillas)

    print()
    print("=" * 90)
    print("SELECCIÓN INTERNA DE MODELOS - SIN USAR EL TEST EXTERNO")
    print("=" * 90)

    for numero, (nombre, plantilla) in enumerate(plantillas.items(), start=1):
        print(f"[{numero:02d}/{total:02d}] {nombre}")
        try:
            modelo = clone(plantilla)
            ejecutar_con_progreso(
                f"{nombre} - entrenamiento interno",
                lambda m=modelo: m.fit(X_train, y_train),
            )
            pred = modelo.predict(X_val)
            prob = probabilidades_modelo(modelo, X_val)
            fila = calcular_metricas_modelo(nombre, y_val, pred, prob)
            resultados.append(fila)
            print(
                f"  AUC {fila['roc_auc']:.4f} | MCC {fila['mcc']:.4f} | "
                f"Score {fila['score_modelo']:.6f} | {fila['motivo']}"
            )
        except Exception as exc:
            print(f"  ERROR seleccionando {nombre}: {exc}")

    metricas = pd.DataFrame(resultados)
    if metricas.empty:
        raise RuntimeError("No se pudo validar ningún modelo en la ventana interna.")

    metricas.sort_values(
        ["elegible", "score_modelo", "roc_auc", "brier_score"],
        ascending=[False, False, False, True],
        inplace=True,
    )
    metricas.reset_index(drop=True, inplace=True)
    metricas["ranking"] = metricas.index + 1

    elegibles = metricas[metricas["elegible"]].head(TOP_MODELOS).copy()
    modo = "CRITERIOS_V3"
    if elegibles.empty:
        # El backtest no debe inventar fiabilidad. Permite continuar únicamente
        # como diagnóstico y deja el fallback explícito en todos los reportes.
        elegibles = metricas.head(TOP_MODELOS).copy()
        modo = "FALLBACK_DIAGNOSTICO_SIN_MODELOS_ELEGIBLES"

    suma = float(elegibles["score_modelo"].clip(lower=0).sum())
    if suma <= 0:
        elegibles["peso"] = 1.0 / len(elegibles)
    else:
        elegibles["peso"] = elegibles["score_modelo"].clip(lower=0) / suma

    metricas["seleccionado"] = metricas["modelo"].isin(elegibles["modelo"])
    metricas["peso_backtest"] = metricas["modelo"].map(
        elegibles.set_index("modelo")["peso"]
    ).fillna(0.0)
    metricas["modo_seleccion"] = modo
    return metricas, elegibles, plantillas, modo


def reentrenar_y_predecir(outer_train, outer_test, features, elegibles, plantillas):
    X_train = limpiar_X(outer_train, features)
    y_train = outer_train[OBJETIVO].astype(int)
    X_test = limpiar_X(outer_test, features)
    scored = outer_test.copy()

    print()
    print("=" * 90)
    print("REENTRENAMIENTO EXTERNO Y PREDICCIONES OOS")
    print("=" * 90)

    for numero, row in enumerate(elegibles.itertuples(index=False), start=1):
        nombre = row.modelo
        print(f"[{numero:02d}/{len(elegibles):02d}] {nombre} | peso {row.peso:.4f}")
        modelo = clone(plantillas[nombre])
        ejecutar_con_progreso(
            f"{nombre} - reentrenamiento externo",
            lambda m=modelo: m.fit(X_train, y_train),
        )
        scored[f"Prob_{nombre}"] = probabilidades_modelo(modelo, X_test)

    return scored


def consenso_historico(scored, elegibles):
    columnas_prob = [f"Prob_{nombre}" for nombre in elegibles["modelo"]]
    pesos = elegibles.set_index("modelo")["peso"]
    pesos_array = np.array([
        pesos.loc[col.replace("Prob_", "", 1)] for col in columnas_prob
    ], dtype=float)
    pesos_array /= pesos_array.sum()

    probs = scored[columnas_prob].to_numpy(dtype=float)
    scored["Consensus_Probability"] = np.average(probs, axis=1, weights=pesos_array)

    relativos = pd.DataFrame(index=scored.index)
    for col in columnas_prob:
        relativos[col] = scored.groupby("Date")[col].rank(pct=True) * 100.0

    matriz_rel = relativos[columnas_prob].to_numpy(dtype=float)
    scored["Score_Relativo"] = np.average(matriz_rel, axis=1, weights=pesos_array)
    scored["Score_Absoluto"] = scored["Consensus_Probability"] * 100.0
    scored["Consensus_Score"] = (
        scored["Score_Relativo"] * 0.70 + scored["Score_Absoluto"] * 0.30
    )
    scored["Model_Dispersion"] = np.std(matriz_rel, axis=1)
    scored["Uncertainty_Score"] = scored["Model_Dispersion"]
    scored["Agreement_Score"] = 100.0 - scored["Uncertainty_Score"]

    scored["Buy_Votes"] = 0
    scored["Neutral_Votes"] = 0
    scored["Sell_Votes"] = 0
    for _, idx in scored.groupby("Date").groups.items():
        posiciones = scored.index.get_indexer(idx)
        matriz_fecha = matriz_rel[posiciones, :]
        q33, q66 = np.percentile(matriz_fecha, [33, 66], axis=0)
        compras = (matriz_fecha >= q66).sum(axis=1)
        ventas = (matriz_fecha <= q33).sum(axis=1)
        neutrales = matriz_fecha.shape[1] - compras - ventas
        scored.loc[idx, "Buy_Votes"] = compras
        scored.loc[idx, "Sell_Votes"] = ventas
        scored.loc[idx, "Neutral_Votes"] = neutrales

    scored["Models_Used"] = len(columnas_prob)
    return scored


def agregar_retornos_futuros(df):
    df = df.sort_values(["Ticker", "Date"]).copy()
    df["Close_Entrada_T1"] = df.groupby("Ticker")["Close"].shift(-1)
    df["Close_Salida_T5"] = df.groupby("Ticker")["Close"].shift(-HORIZONTE_DIAS)
    df["Retorno_Futuro"] = (
        df["Close_Salida_T5"] / df["Close_Entrada_T1"] - 1.0
    )
    return df.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def construir_carteras(scored):
    fechas = sorted(scored["Date"].dropna().unique())[::HORIZONTE_DIAS]
    carteras = []
    operaciones = []

    for fecha in fechas:
        dia = scored[scored["Date"] == fecha].dropna(
            subset=["Score_Final", "Retorno_Futuro"]
        ).copy()
        if dia.empty:
            continue
        seleccion = (
            dia[dia["Señal"].isin(SENALES_OPERABLES)]
            .sort_values("Score_Final", ascending=False)
            .head(TOP_N)
        )
        if seleccion.empty:
            carteras.append({
                "Fecha": fecha,
                "Tickers": "CASH",
                "Score_Usado": "Score_Final_IA_OOS",
                "Scores": "",
                "Probabilidades": "",
                "Cantidad_Activos": 0,
                "Retorno_Bruto_5d": 0.0,
                "Costo_Ida_Vuelta": 0.0,
                "Retorno_Neto_5d": 0.0,
                "Estado": "CASH_SIN_SEÑAL",
            })
            continue
        retorno_bruto = float(seleccion["Retorno_Futuro"].mean())
        retorno_neto = retorno_bruto - COSTO_IDA_VUELTA
        carteras.append({
            "Fecha": fecha,
            "Tickers": " | ".join(seleccion["Ticker"].astype(str)),
            "Score_Usado": "Score_Final_IA_OOS",
            "Scores": " | ".join(f"{x:.4f}" for x in seleccion["Score_Final"]),
            "Probabilidades": " | ".join(
                f"{x:.4f}" for x in seleccion["Consensus_Probability"]
            ),
            "Cantidad_Activos": len(seleccion),
            "Retorno_Bruto_5d": retorno_bruto,
            "Costo_Ida_Vuelta": COSTO_IDA_VUELTA,
            "Retorno_Neto_5d": retorno_neto,
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
            })

    return pd.DataFrame(carteras), pd.DataFrame(operaciones)


def calcular_equity(carteras):
    capital = CAPITAL_INICIAL
    filas = []
    for _, row in carteras.iterrows():
        anterior = capital
        capital *= 1.0 + float(row["Retorno_Neto_5d"])
        filas.append({
            "Fecha": row["Fecha"],
            "Capital_Inicial_Periodo": anterior,
            "Retorno_Neto_5d": row["Retorno_Neto_5d"],
            "Capital_Final_Periodo": capital,
        })
    return pd.DataFrame(filas)


def calcular_metricas(carteras, equity, metadata):
    if carteras.empty or equity.empty:
        raise RuntimeError("El período OOS no produjo carteras evaluables.")
    operadas = carteras[carteras["Cantidad_Activos"] > 0]
    r = operadas["Retorno_Neto_5d"].dropna()
    ganadoras = r[r > 0]
    perdedoras = r[r <= 0]
    capital_final = float(equity["Capital_Final_Periodo"].iloc[-1])
    running_max = equity["Capital_Final_Periodo"].cummax()
    drawdown = equity["Capital_Final_Periodo"] / running_max - 1.0
    dias = max((pd.Timestamp(carteras["Fecha"].max()) - pd.Timestamp(carteras["Fecha"].min())).days, 1)
    years = dias / 365.25
    profit_factor = ganadoras.sum() / abs(perdedoras.sum()) if len(r) and abs(perdedoras.sum()) > 0 else np.nan
    sharpe = r.mean() / r.std() * np.sqrt(252 / HORIZONTE_DIAS) if len(r) > 1 and r.std() > 0 else np.nan
    downside = r[r < 0].std()
    sortino = r.mean() / downside * np.sqrt(252 / HORIZONTE_DIAS) if pd.notna(downside) and downside > 0 else np.nan

    return {
        "tipo_backtest": "ENSEMBLE_IA_ANIDADO_OOS",
        "periodos_evaluados": len(carteras),
        "carteras_operadas": len(r),
        "periodos_cash": int((carteras["Cantidad_Activos"] == 0).sum()),
        "participacion_mercado_pct": round(len(r) / len(carteras) * 100, 2),
        "top_activos_por_fecha": TOP_N,
        "horizonte_dias": HORIZONTE_DIAS,
        "costo_por_lado": COSTO_POR_LADO,
        "costo_ida_vuelta": COSTO_IDA_VUELTA,
        "ganadoras": int((r > 0).sum()),
        "perdedoras": int((r <= 0).sum()),
        "win_rate": round((r > 0).mean() * 100, 2) if len(r) else 0.0,
        "retorno_promedio_5d": round(r.mean() * 100, 2) if len(r) else 0.0,
        "ganancia_promedio": round(ganadoras.mean() * 100, 2) if len(ganadoras) else 0.0,
        "perdida_promedio": round(perdedoras.mean() * 100, 2) if len(perdedoras) else 0.0,
        "profit_factor": round(float(profit_factor), 3) if pd.notna(profit_factor) else None,
        "sharpe_aprox": round(float(sharpe), 3) if pd.notna(sharpe) else None,
        "sortino_aprox": round(float(sortino), 3) if pd.notna(sortino) else None,
        "capital_inicial": CAPITAL_INICIAL,
        "capital_final": round(capital_final, 2),
        "retorno_total": round((capital_final / CAPITAL_INICIAL - 1.0) * 100, 2),
        "cagr": round(((capital_final / CAPITAL_INICIAL) ** (1 / years) - 1) * 100, 2),
        "max_drawdown": round(float(drawdown.min()) * 100, 2),
        "score_usado": "Score_Final_IA_OOS",
        "politica_operativa": "SOLO_COMPRA_Y_COMPRA_FUERTE; CASH_EN_OTRO_CASO",
        **metadata,
    }


def exportar(market, metricas, metricas_modelos, scored, carteras, operaciones, equity):
    out = ROOT / "reportes" / market.market
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "scores_oos": out / f"backtest_ai_scores_{market.market}.csv",
        "modelos": out / f"backtest_ai_modelos_{market.market}.csv",
        "carteras": out / f"backtest_carteras_{market.market}.csv",
        "operaciones": out / f"backtest_operaciones_{market.market}.csv",
        "equity": out / f"backtest_equity_{market.market}.csv",
        "resumen": out / f"backtest_resumen_{market.market}.csv",
        "excel": out / f"backtest_market_ai_{market.market}.xlsx",
        "txt": out / f"backtest_market_ai_{market.market}.txt",
    }
    scored.to_csv(paths["scores_oos"], index=False, encoding="utf-8-sig")
    metricas_modelos.to_csv(paths["modelos"], index=False, encoding="utf-8-sig")
    carteras.to_csv(paths["carteras"], index=False, encoding="utf-8-sig")
    operaciones.to_csv(paths["operaciones"], index=False, encoding="utf-8-sig")
    equity.to_csv(paths["equity"], index=False, encoding="utf-8-sig")
    pd.DataFrame([metricas]).to_csv(paths["resumen"], index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(paths["excel"], engine="openpyxl") as writer:
        pd.DataFrame([metricas]).to_excel(writer, index=False, sheet_name="Resumen")
        metricas_modelos.to_excel(writer, index=False, sheet_name="Modelos")
        equity.to_excel(writer, index=False, sheet_name="Equity")
        carteras.to_excel(writer, index=False, sheet_name="Carteras")
        operaciones.to_excel(writer, index=False, sheet_name="Operaciones")
        scored.to_excel(writer, index=False, sheet_name="Scores_OOS")
    with open(paths["txt"], "w", encoding="utf-8") as f:
        f.write(f"MARKET AI BACKTEST ANIDADO OOS - {market.display_name}\n")
        f.write("=" * 90 + "\n\n")
        for key, value in metricas.items():
            f.write(f"{key:32}: {value}\n")
        f.write("\nMODELOS DE LA VALIDACIÓN INTERNA\n")
        f.write(metricas_modelos.to_string(index=False))
    return paths


def ejecutar_backtest(mercado="merval"):
    inicio = datetime.now()
    market = MarketConfig(mercado)
    print("=" * 90)
    print(f"MARKET AI BACKTEST ANIDADO OOS - {market.display_name}")
    print("=" * 90)

    df = cargar_dataset(market)
    features = columnas_features(df)
    outer_train, outer_test, meta_outer = crear_split_temporal(
        df, test_size=TEST_EXTERNO, purga_ruedas=PURGA_RUEDAS
    )
    inner_train, inner_val, meta_inner = crear_split_temporal(
        outer_train, test_size=VALIDACION_INTERNA, purga_ruedas=PURGA_RUEDAS
    )

    print(f"Dataset IA             : {market.dataset_csv}")
    print(f"Registros              : {len(df):,}")
    print(f"Variables              : {len(features)}")
    print(f"Selección interna      : {meta_inner['train_fecha_min']} -> {meta_inner['test_fecha_max']}")
    print(f"Test exterior intacto  : {meta_outer['test_fecha_min']} -> {meta_outer['test_fecha_max']}")
    print(f"Purga por frontera     : {PURGA_RUEDAS} ruedas por ticker")

    metricas_modelos, elegibles, plantillas, modo = seleccionar_modelos(
        inner_train, inner_val, features
    )
    scored = reentrenar_y_predecir(
        outer_train, outer_test, features, elegibles, plantillas
    )
    scored = consenso_historico(scored, elegibles)
    defaults_tecnicos = {
        "RSI_14": 50.0,
        "MACD": 0.0,
        "MACD_Signal": 0.0,
        "Volume_Ratio": 1.0,
        "Return_20d": 0.0,
        "EMA_20": scored["Close"],
        "EMA_50": scored["Close"],
        "EMA_200": scored["Close"],
    }
    for columna, valor in defaults_tecnicos.items():
        if columna not in scored.columns:
            scored[columna] = valor
    scored = calcular_score_tecnico(scored)
    tasa_base = float(inner_val[OBJETIVO].astype(int).mean())
    scored = clasificar_senal(scored, tasa_base)
    scored = agregar_retornos_futuros(scored)

    carteras, operaciones = construir_carteras(scored)
    equity = calcular_equity(carteras)
    seleccion = ", ".join(
        f"{row.modelo}:{row.peso:.4f}" for row in elegibles.itertuples(index=False)
    )
    metadata = {
        "modo_seleccion": modo,
        "modelos_usados": seleccion,
        "tasa_base_interna": round(tasa_base, 6),
        "train_externo_desde": meta_outer["train_fecha_min"],
        "train_externo_hasta": meta_outer["train_fecha_max"],
        "test_oos_desde": meta_outer["test_fecha_min"],
        "test_oos_hasta": meta_outer["test_fecha_max"],
        "separacion_temporal_estricta": True,
        "pesos_ajustados_sin_test_externo": True,
    }
    metricas = calcular_metricas(carteras, equity, metadata)
    paths = exportar(
        market, metricas, metricas_modelos, scored, carteras, operaciones, equity
    )

    print()
    print("=" * 90)
    print("RESULTADOS BACKTEST IA ANIDADO OOS")
    print("=" * 90)
    for key, value in metricas.items():
        print(f"{key:32}: {value}")
    print()
    print("ARCHIVOS GENERADOS")
    for path in paths.values():
        print(path)
    print(f"Tiempo total                    : {datetime.now() - inicio}")
    return metricas, carteras, operaciones, equity


if __name__ == "__main__":
    mercado_arg = sys.argv[1].lower() if len(sys.argv) > 1 else "merval"
    ejecutar_backtest(mercado_arg)
