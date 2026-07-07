import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from config import MarketConfig

try:
    from source.investment_policy import obtener_politica
except ModuleNotFoundError:
    from investment_policy import obtener_politica


# =============================================================================
# DECISION ENGINE MULTI-MERCADO
# =============================================================================

def rutas_decision_engine(nombre_mercado):
    market = MarketConfig(nombre_mercado)
    reportes_dir = ROOT / "reportes" / market.market

    return {
        "market": market,
        "reportes_dir": reportes_dir,
        "ranking": reportes_dir / "ranking_diario.csv",
        "regime": reportes_dir / f"market_regime_{market.market}_v2.csv",
        "csv_out": reportes_dir / f"decision_engine_{market.market}.csv",
        "xlsx_out": reportes_dir / f"decision_engine_{market.market}.xlsx",
        "txt_out": reportes_dir / f"decision_engine_{market.market}.txt",
    }


def verificar_archivos(rutas):
    if not rutas["ranking"].exists():
        raise FileNotFoundError(f"No existe ranking diario: {rutas['ranking']}")

    if not rutas["regime"].exists():
        raise FileNotFoundError(f"No existe régimen de mercado: {rutas['regime']}")


def cargar_ranking(rutas):
    return pd.read_csv(rutas["ranking"])


def cargar_regimen(rutas):
    return pd.read_csv(rutas["regime"]).iloc[0]


def obtener_columna_probabilidad(ranking):
    if "Consensus_Probability" in ranking.columns:
        return "Consensus_Probability"

    if "Prob_IA" in ranking.columns:
        return "Prob_IA"

    raise ValueError("No se encontró columna de probabilidad en ranking.")


def obtener_columna_score(ranking):
    if "Score_Final" in ranking.columns:
        return "Score_Final"

    raise ValueError("No se encontró columna Score_Final en ranking.")


def calcular_market_ai_index(ranking, regimen, politica):
    prob_col = obtener_columna_probabilidad(ranking)
    score_col = obtener_columna_score(ranking)

    prob_media = ranking[prob_col].mean()
    score_medio = ranking[score_col].mean()

    compras = ranking["Señal"].isin(["COMPRA", "COMPRA FUERTE"]).sum()
    evitar = (ranking["Señal"] == "EVITAR").sum()
    total = len(ranking)

    compras_pct = compras / total if total else 0
    evitar_pct = evitar / total if total else 0

    indice = (
        score_medio * 0.45
        + (prob_media * 100) * 0.25
        + compras_pct * 20
        - evitar_pct * 15
        - politica["nivel_riesgo"] * 0.15
    )

    indice = max(0, min(100, indice))

    return round(indice, 2)


def obtener_semaforo(indice):
    if indice >= 80:
        return "🟢", "EXCELENTE"

    if indice >= 60:
        return "🟡", "FAVORABLE"

    if indice >= 40:
        return "🟠", "PRECAUCIÓN"

    return "🔴", "RIESGO ALTO"
def obtener_mejor_activo(ranking):
    score_col = obtener_columna_score(ranking)
    prob_col = obtener_columna_probabilidad(ranking)

    fila = ranking.sort_values(
        score_col,
        ascending=False
    ).iloc[0]

    nombre = fila["Nombre"] if "Nombre" in ranking.columns else fila["Ticker"]

    return {
        "ticker": fila["Ticker"],
        "nombre": nombre,
        "probabilidad": round(float(fila[prob_col]) * 100, 2),
        "score_final": round(float(fila[score_col]), 2),
        "senal": fila["Señal"] if "Señal" in ranking.columns else "",
    }


def evaluar_activos_aptos(ranking, politica):
    prob_col = obtener_columna_probabilidad(ranking)
    score_col = obtener_columna_score(ranking)

    if "Señal" not in ranking.columns:
        return pd.DataFrame()

    candidatos = ranking[
        (ranking[prob_col] >= politica["min_prob_compra"])
        &
        (ranking[score_col] >= politica["min_score_final"])
        &
        (ranking["Señal"].isin(["COMPRA", "COMPRA FUERTE", "VIGILAR"]))
    ].copy()

    return candidatos


def construir_justificacion(
    market,
    regimen,
    politica,
    ranking,
    candidatos,
    indice
):
    prob_col = obtener_columna_probabilidad(ranking)
    score_col = obtener_columna_score(ranking)

    prob_media = ranking[prob_col].mean()
    score_medio = ranking[score_col].mean()
    compras = ranking["Señal"].isin(["COMPRA", "COMPRA FUERTE"]).sum()
    evitar = (ranking["Señal"] == "EVITAR").sum()

    texto = []

    texto.append(
        f"Mercado analizado: {market.display_name}."
    )

    texto.append(
        f"Régimen detectado: {regimen['Regimen']}."
    )

    texto.append(
        politica["descripcion"]
    )

    texto.append(
        f"Nivel de riesgo de la política: {politica['nivel_riesgo']}/100."
    )

    texto.append(
        f"Market AI Index: {indice}/100."
    )

    texto.append(
        f"Activos analizados: {len(ranking)}."
    )

    texto.append(
        f"Probabilidad media: {prob_media:.2%}."
    )

    texto.append(
        f"Score medio: {score_medio:.2f}."
    )

    texto.append(
        f"Señales de compra: {compras}."
    )

    texto.append(
        f"Señales de evitar: {evitar}."
    )

    texto.append(
        f"Activos aptos según política: {len(candidatos)}."
    )

    if len(candidatos) == 0:
        texto.append(
            "Ningún activo supera simultáneamente los mínimos exigidos por la política."
        )
    else:
        texto.append(
            "Existen activos que cumplen las condiciones mínimas de inversión."
        )

    return " ".join(texto)


def generar_decision(nombre_mercado="merval"):
    inicio = datetime.now()

    rutas = rutas_decision_engine(nombre_mercado)
    market = rutas["market"]

    print("=" * 80)
    print(f"DECISION ENGINE - {market.display_name}")
    print("=" * 80)

    verificar_archivos(rutas)

    ranking = cargar_ranking(rutas)
    regimen = cargar_regimen(rutas)

    politica = obtener_politica(regimen["Regimen"])

    candidatos = evaluar_activos_aptos(
        ranking,
        politica
    )

    indice = calcular_market_ai_index(
        ranking,
        regimen,
        politica
    )

    emoji, estado = obtener_semaforo(indice)

    mejor = obtener_mejor_activo(ranking)

    justificacion = construir_justificacion(
        market=market,
        regimen=regimen,
        politica=politica,
        ranking=ranking,
        candidatos=candidatos,
        indice=indice
    )

    resultado = {
        "Fecha": regimen["Fecha"],
        "Mercado": market.market,
        "Mercado_Display": market.display_name,
        "Regimen": regimen["Regimen"],
        "Market_AI_Index": indice,
        "Semaforo": emoji,
        "Estado": estado,
        "Nivel_Riesgo": politica["nivel_riesgo"],
        "Decision": politica["accion_principal"],
        "Cash_pct": politica["cash_pct"],
        "Acciones_pct": politica["acciones_pct"],
        "Max_Posiciones": politica["max_posiciones"],
        "Stop_Loss": politica["stop_loss"],
        "Take_Profit": politica["take_profit"],
        "Rebalanceo": politica["rebalanceo"],
        "Activos_Aptos": len(candidatos),
        "Mejor_Ticker": mejor["ticker"],
        "Mejor_Activo": mejor["nombre"],
        "Probabilidad_IA": mejor["probabilidad"],
        "Score_Final": mejor["score_final"],
        "Señal": mejor["senal"],
        "Justificacion": justificacion,
    }
    df_resultado = pd.DataFrame([resultado])

    df_resultado.to_csv(
        rutas["csv_out"],
        index=False,
        encoding="utf-8-sig"
    )

    df_resultado.to_excel(
        rutas["xlsx_out"],
        index=False
    )

    with open(rutas["txt_out"], "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"MARKET AI DECISION ENGINE - {market.display_name}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Fecha                 : {resultado['Fecha']}\n")
        f.write(f"Mercado               : {resultado['Mercado_Display']}\n")
        f.write(f"Régimen               : {resultado['Regimen']}\n")
        f.write(f"Market AI Index       : {resultado['Market_AI_Index']}/100\n")
        f.write(f"Semáforo              : {resultado['Semaforo']} {resultado['Estado']}\n")
        f.write(f"Nivel Riesgo          : {resultado['Nivel_Riesgo']}/100\n\n")

        f.write("-" * 80 + "\n")
        f.write("POLÍTICA APLICADA\n")
        f.write("-" * 80 + "\n")

        f.write(f"Decisión              : {resultado['Decision']}\n")
        f.write(f"Cash recomendado      : {resultado['Cash_pct']} %\n")
        f.write(f"Capital en activos    : {resultado['Acciones_pct']} %\n")
        f.write(f"Máximo posiciones     : {resultado['Max_Posiciones']}\n")
        f.write(f"Stop Loss             : {resultado['Stop_Loss']} %\n")
        f.write(f"Take Profit           : {resultado['Take_Profit']} %\n")
        f.write(f"Rebalanceo            : {resultado['Rebalanceo']}\n\n")

        f.write("-" * 80 + "\n")
        f.write("MEJOR ACTIVO RELATIVO\n")
        f.write("-" * 80 + "\n")

        f.write(f"Ticker                : {resultado['Mejor_Ticker']}\n")
        f.write(f"Nombre                : {resultado['Mejor_Activo']}\n")
        f.write(f"Probabilidad IA       : {resultado['Probabilidad_IA']} %\n")
        f.write(f"Score Final           : {resultado['Score_Final']}\n")
        f.write(f"Señal                 : {resultado['Señal']}\n")
        f.write(f"Activos aptos         : {resultado['Activos_Aptos']}\n\n")

        f.write("-" * 80 + "\n")
        f.write("JUSTIFICACIÓN\n")
        f.write("-" * 80 + "\n")
        f.write(resultado["Justificacion"])
        f.write("\n")

    print()
    print("=" * 80)
    print("DECISIÓN DEL SISTEMA")
    print("=" * 80)

    print(f"Fecha                : {resultado['Fecha']}")
    print(f"Mercado              : {resultado['Mercado_Display']}")
    print(f"Régimen              : {resultado['Regimen']}")
    print(f"Market AI Index      : {resultado['Market_AI_Index']}")
    print(f"Semáforo             : {resultado['Semaforo']} {resultado['Estado']}")
    print()
    print(f"DECISIÓN             : {resultado['Decision']}")
    print(f"Cash                 : {resultado['Cash_pct']} %")
    print(f"Activos              : {resultado['Acciones_pct']} %")
    print(f"Máx. posiciones      : {resultado['Max_Posiciones']}")
    print()
    print("MEJOR ACTIVO")
    print("-" * 40)
    print(f"{resultado['Mejor_Ticker']} - {resultado['Mejor_Activo']}")
    print(f"Probabilidad IA      : {resultado['Probabilidad_IA']} %")
    print(f"Score Final          : {resultado['Score_Final']}")
    print(f"Señal                : {resultado['Señal']}")
    print(f"Activos Aptos        : {resultado['Activos_Aptos']}")
    print()
    print("JUSTIFICACIÓN")
    print("-" * 40)
    print(resultado["Justificacion"])
    print()

    print("=" * 80)
    print("REPORTES GENERADOS")
    print("=" * 80)
    print(rutas["csv_out"])
    print(rutas["xlsx_out"])
    print(rutas["txt_out"])

    print()
    print("=" * 80)
    print("DECISION ENGINE FINALIZADO")
    print("=" * 80)
    print(f"Tiempo total : {datetime.now() - inicio}")

    return df_resultado


if __name__ == "__main__":
    generar_decision("crypto")