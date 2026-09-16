# portfolio_optimizer.py

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from config import MarketConfig


CAPITAL_INICIAL = 100000.0

MAX_PESO_ACTIVO = 45.0
MIN_PESO_ACTIVO = 8.0
TOP_CANDIDATOS_EXTRA = 8


def rutas_portfolio(nombre_mercado):
    market = MarketConfig(nombre_mercado)
    reportes_dir = ROOT / "reportes" / market.market

    return {
        "market": market,
        "ranking": reportes_dir / "ranking_diario.csv",
        "decision": reportes_dir / f"decision_engine_{market.market}.csv",
        "csv_out": reportes_dir / f"portfolio_optimizer_{market.market}.csv",
        "xlsx_out": reportes_dir / f"portfolio_optimizer_{market.market}.xlsx",
        "txt_out": reportes_dir / f"portfolio_optimizer_{market.market}.txt",
    }


def verificar_archivos(rutas):
    if not rutas["ranking"].exists():
        raise FileNotFoundError(f"No existe ranking diario: {rutas['ranking']}")

    if not rutas["decision"].exists():
        raise FileNotFoundError(f"No existe Decision Engine: {rutas['decision']}")


def cargar_decision(rutas):
    return pd.read_csv(rutas["decision"]).iloc[0]


def cargar_ranking(rutas):
    ranking = pd.read_csv(rutas["ranking"])

    if "Score_Final" not in ranking.columns:
        raise ValueError("No existe Score_Final en ranking.")

    return ranking


def normalizar_serie(s):
    s = pd.to_numeric(s, errors="coerce").fillna(0)

    minimo = s.min()
    maximo = s.max()

    if maximo - minimo == 0:
        return pd.Series(0.5, index=s.index)

    return (s - minimo) / (maximo - minimo)


def obtener_probabilidad(df):
    if "Consensus_Probability" in df.columns:
        prob = pd.to_numeric(df["Consensus_Probability"], errors="coerce").fillna(0)

        if prob.max() > 1:
            prob = prob / 100

        return prob.clip(0, 1)

    if "Probabilidad_IA" in df.columns:
        prob = pd.to_numeric(df["Probabilidad_IA"], errors="coerce").fillna(0)

        if prob.max() > 1:
            prob = prob / 100

        return prob.clip(0, 1)

    return pd.Series(0.5, index=df.index)


def obtener_riesgo(df):
    posibles = [
        "Volatilidad",
        "Volatility",
        "Riesgo",
        "Risk",
        "Drawdown",
        "Max_Drawdown",
    ]

    for col in posibles:
        if col in df.columns:
            return normalizar_serie(df[col])

    return pd.Series(0.5, index=df.index)


def obtener_momentum(df):
    posibles = [
        "Retorno_20d",
        "Return_20d",
        "Momentum",
        "Score_Tecnico",
        "Consensus_Score",
    ]

    for col in posibles:
        if col in df.columns:
            return normalizar_serie(df[col])

    return normalizar_serie(df["Score_Final"])


def factor_senal(senal):
    mapa = {
        "COMPRA FUERTE": 1.25,
        "COMPRA": 1.10,
        "VIGILAR": 0.90,
        "NEUTRAL": 0.60,
        "VENTA": 0.25,
        "VENTA FUERTE": 0.10,
    }

    return mapa.get(str(senal).strip().upper(), 0.50)


def seleccionar_activos(decision, ranking):
    max_posiciones = int(decision.get("Max_Posiciones", 0))
    activos_aptos = int(decision.get("Activos_Aptos", 0))

    # Si Decision Engine determina que no hay activos aptos,
    # Portfolio Optimizer no debe forzar posiciones.
    if max_posiciones <= 0 or activos_aptos <= 0:
        return pd.DataFrame()

    ranking = ranking.copy()

    if "Señal" in ranking.columns:
        ranking = ranking[
            ranking["Señal"].isin(["COMPRA FUERTE", "COMPRA"])
        ].copy()

    if ranking.empty:
        return ranking

    ranking["score_norm"] = normalizar_serie(ranking["Score_Final"])
    ranking["prob_norm"] = obtener_probabilidad(ranking)
    ranking["riesgo_norm"] = obtener_riesgo(ranking)
    ranking["momentum_norm"] = obtener_momentum(ranking)

    if "Señal" in ranking.columns:
        ranking["factor_senal"] = ranking["Señal"].apply(factor_senal)
    else:
        ranking["factor_senal"] = 0.75

    ranking["score_ajustado"] = (
        0.45 * ranking["score_norm"]
        + 0.25 * ranking["prob_norm"]
        + 0.20 * ranking["momentum_norm"]
        + 0.10 * ranking["factor_senal"]
        - 0.20 * ranking["riesgo_norm"]
    )

    ranking["score_ajustado"] = ranking["score_ajustado"].clip(lower=0)

    limite = min(max_posiciones, activos_aptos)

    candidatos = ranking.sort_values("score_ajustado", ascending=False).head(
        max(limite, TOP_CANDIDATOS_EXTRA)
    )

    return candidatos.head(limite).copy()


def limitar_concentracion(pesos):
    pesos = pesos.copy()

    for _ in range(10):
        exceso = pesos[pesos > MAX_PESO_ACTIVO] - MAX_PESO_ACTIVO

        if exceso.sum() <= 0:
            break

        pesos[pesos > MAX_PESO_ACTIVO] = MAX_PESO_ACTIVO

        redistribuir = exceso.sum()
        libres = pesos[pesos < MAX_PESO_ACTIVO]

        if libres.empty:
            break

        pesos.loc[libres.index] += redistribuir * libres / libres.sum()

    pesos = pesos / pesos.sum() * 100

    return pesos


def calcular_pesos(cartera):
    if cartera.empty:
        return cartera

    cartera = cartera.copy()

    base = cartera["score_ajustado"].clip(lower=0)

    if base.sum() == 0:
        cartera["Porcentaje"] = 100 / len(cartera)
    else:
        cartera["Porcentaje"] = base / base.sum() * 100


    if len(cartera) > 2:
        cartera["Porcentaje"] = limitar_concentracion(cartera["Porcentaje"])

    if len(cartera) > 3:
        cartera.loc[cartera["Porcentaje"] < MIN_PESO_ACTIVO, "Porcentaje"] = MIN_PESO_ACTIVO
        cartera["Porcentaje"] = cartera["Porcentaje"] / cartera["Porcentaje"].sum() * 100

    return cartera


def distribuir_capital(decision, cartera):
    cash_pct = float(decision.get("Cash_pct", 100))
    activos_pct = float(decision.get("Acciones_pct", 0))

    capital_cash = CAPITAL_INICIAL * cash_pct / 100
    capital_activos = CAPITAL_INICIAL * activos_pct / 100

    # Si no hay posiciones, todo el capital queda liquido.
    if cartera.empty:
        return cartera, round(CAPITAL_INICIAL, 2), 0.0

    cartera = cartera.copy()
    cartera["Capital"] = capital_activos * cartera["Porcentaje"] / 100

    return cartera, round(capital_cash, 2), round(capital_activos, 2)


def construir_portafolio(nombre_mercado="merval"):
    inicio = datetime.now()

    rutas = rutas_portfolio(nombre_mercado)
    market = rutas["market"]

    print("=" * 80)
    print(f"PORTFOLIO OPTIMIZER - {market.display_name}")
    print("=" * 80)

    verificar_archivos(rutas)

    decision = cargar_decision(rutas)
    ranking = cargar_ranking(rutas)

    cartera = seleccionar_activos(decision, ranking)
    cartera = calcular_pesos(cartera)
    cartera, cash, invertido = distribuir_capital(decision, cartera)

    columnas_exportar = [
        "Ticker",
        "Nombre",
        "Close",
        "Consensus_Probability",
        "Consensus_Score",
        "Score_Tecnico",
        "Score_Final",
        "score_ajustado",
        "riesgo_norm",
        "momentum_norm",
        "SeÃ±al",
        "Porcentaje",
        "Capital",
    ]

    columnas_exportar = [c for c in columnas_exportar if c in cartera.columns]

    if cartera.empty:
        exportar = pd.DataFrame(columns=[
            "Ticker",
            "Nombre",
            "Score_Final",
            "SeÃ±al",
            "Porcentaje",
            "Capital",
        ])
    else:
        exportar = cartera[columnas_exportar].copy()

    exportar.to_csv(rutas["csv_out"], index=False, encoding="utf-8-sig")
    exportar.to_excel(rutas["xlsx_out"], index=False)

    with open(rutas["txt_out"], "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"MARKET AI PORTFOLIO OPTIMIZER - {market.display_name}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Fecha                 : {decision.get('Fecha', '')}\n")
        f.write(f"Mercado               : {market.display_name}\n")
        f.write(f"RÃ©gimen               : {decision.get('Regimen', '')}\n")
        f.write(f"DecisiÃ³n              : {decision.get('Decision', '')}\n")
        f.write(f"Capital Inicial       : ${CAPITAL_INICIAL:,.2f}\n")
        f.write(f"Cash                  : ${cash:,.2f}\n")
        f.write(f"Capital Activos       : ${invertido:,.2f}\n")
        f.write(f"Posiciones            : {len(cartera)}\n\n")

        f.write("-" * 80 + "\n")
        f.write("CARTERA SUGERIDA\n")
        f.write("-" * 80 + "\n")

        if cartera.empty:
            f.write("No se recomienda abrir posiciones.\n")
            f.write("Todo el capital permanece lÃ­quido segÃºn la polÃ­tica aplicada.\n")
        else:
            for _, row in cartera.iterrows():
                f.write(
                    f"{row.get('Ticker', ''):12} | "
                    f"{str(row.get('Nombre', ''))[:30]:30} | "
                    f"{row.get('Porcentaje', 0):6.2f}% | "
                    f"${row.get('Capital', 0):12,.2f} | "
                    f"Score {row.get('Score_Final', 0):6.2f} | "
                    f"Ajustado {row.get('score_ajustado', 0):6.3f} | "
                    f"{row.get('SeÃ±al', '')}\n"
                )

    print()
    print("=" * 80)
    print("PORTAFOLIO SUGERIDO")
    print("=" * 80)

    print(f"Mercado         : {market.display_name}")
    print(f"Capital Inicial : ${CAPITAL_INICIAL:,.2f}")
    print(f"Cash            : ${cash:,.2f}")
    print(f"Invertido       : ${invertido:,.2f}")
    print(f"Posiciones      : {len(cartera)}")
    print()

    if cartera.empty:
        print("No se recomienda abrir posiciones.")
        print("Todo el capital permanece lÃ­quido.")
    else:
        mostrar = [
            "Ticker",
            "Nombre",
            "Porcentaje",
            "Capital",
            "Score_Final",
            "score_ajustado",
            "SeÃ±al",
        ]

        mostrar = [c for c in mostrar if c in cartera.columns]
        print(cartera[mostrar].to_string(index=False))

    print()
    print("=" * 80)
    print("ARCHIVOS GENERADOS")
    print("=" * 80)
    print(rutas["csv_out"])
    print(rutas["xlsx_out"])
    print(rutas["txt_out"])

    print()
    print("=" * 80)
    print("PORTFOLIO OPTIMIZER FINALIZADO")
    print("=" * 80)
    print(f"Tiempo total : {datetime.now() - inicio}")

    return exportar


if __name__ == "__main__":
    construir_portafolio("crypto")
