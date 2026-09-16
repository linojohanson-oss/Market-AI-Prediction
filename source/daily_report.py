import os
from pathlib import Path
from datetime import datetime

import pandas as pd
from pandas.errors import EmptyDataError

ROOT = Path(__file__).resolve().parent.parent


def leer_csv_opcional(ruta: Path) -> pd.DataFrame:
    """
    Lee un CSV opcional.

    Si el archivo no existe o existe pero esta vacio, devuelve un
    DataFrame vacio para que el reporte pueda continuar normalmente.
    """
    if not ruta.exists() or ruta.stat().st_size == 0:
        return pd.DataFrame()

    try:
        return pd.read_csv(ruta)
    except EmptyDataError:
        return pd.DataFrame()


def formatear_indicador(valor, decimales=2):
    """
    Formatea indicadores tecnicos.
    Devuelve NO DISPONIBLE si el valor es nulo o NaN.
    """
    if pd.isna(valor):
        return "NO DISPONIBLE"

    try:
        return f"{float(valor):.{decimales}f}"
    except (TypeError, ValueError):
        return "NO DISPONIBLE"

def generar_daily_report(mercado="merval"):
    reportes_dir = ROOT / "reportes" / mercado

    ranking_path = reportes_dir / "ranking_diario.csv"
    regime_path = reportes_dir / f"market_regime_{mercado}_v2.csv"
    decision_path = reportes_dir / f"decision_engine_{mercado}.csv"
    portfolio_path = reportes_dir / f"portfolio_optimizer_{mercado}.csv"
    indicadores_path = reportes_dir / f"indicadores_resumen_{mercado}.csv"

    output_txt = reportes_dir / f"market_ai_daily_report_{mercado}.txt"

    # Validaciones
    archivos_necesarios = {
        "Ranking": ranking_path,
        "Regimen": regime_path,
        "Decision Engine": decision_path,
    }

    for nombre, ruta in archivos_necesarios.items():
        if not ruta.exists():
            raise FileNotFoundError(
                f"ERROR - Falta archivo requerido ({nombre}): {ruta}"
            )

    ranking = pd.read_csv(ranking_path)
    regime = pd.read_csv(regime_path).iloc[0]
    decision = pd.read_csv(decision_path).iloc[0]

    # Portfolio e indicadores son opcionales.
    # Un archivo vacio es un estado valido, no un error.
    portfolio = leer_csv_opcional(portfolio_path)
    indicadores = leer_csv_opcional(indicadores_path)

    top = ranking.head(10)

    # ============================================================
    # GENERAR REPORTE
    # ============================================================
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("MARKET AI DAILY REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Fecha                 : {regime.get('Fecha', '')}\n")
        f.write(f"Mercado               : {mercado.upper()}\n")
        f.write(f"Generado              : {datetime.now()}\n\n")

        # ------------------------------------------------------------
        # REGIMEN DE MERCADO
        # ------------------------------------------------------------
        f.write("-" * 80 + "\n")
        f.write("REGIMEN DE MERCADO\n")
        f.write("-" * 80 + "\n")
        f.write(f"Regimen               : {regime.get('Regimen', '')}\n")
        f.write(
            f"Accion recomendada    : "
            f"{regime.get('Accion_Recomendada', '')}\n"
        )
        f.write(
            f"Descripcion           : "
            f"{regime.get('Descripcion', '')}\n"
        )
        f.write(
            f"Probabilidad media    : "
            f"{regime.get('Prob_Media', 0):.4f}\n"
        )
        f.write(
            f"Probabilidad maxima   : "
            f"{regime.get('Prob_Max', 0):.4f}\n"
        )
        f.write(
            f"Score medio           : "
            f"{regime.get('Score_Medio', 0):.2f}\n\n"
        )

        # ------------------------------------------------------------
        # DECISION IA
        # ------------------------------------------------------------
        f.write("-" * 80 + "\n")
        f.write("DECISION IA\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"Decision              : "
            f"{decision.get('Decision', '')}\n"
        )
        f.write(
            f"Market AI Index       : "
            f"{decision.get('Market_AI_Index', '')}\n"
        )
        f.write(
            f"Semaforo              : "
            f"{decision.get('Semaforo', '')} "
            f"{decision.get('Estado', '')}\n"
        )
        f.write(
            f"Cash recomendado      : "
            f"{decision.get('Cash_pct', '')} %\n"
        )
        f.write(
            f"Acciones              : "
            f"{decision.get('Acciones_pct', '')} %\n"
        )
        f.write(
            f"Maximo posiciones     : "
            f"{decision.get('Max_Posiciones', '')}\n\n"
        )

        # ------------------------------------------------------------
        # TOP 10 RANKING
        # ------------------------------------------------------------
        f.write("-" * 80 + "\n")
        f.write("TOP 10 RANKING\n")
        f.write("-" * 80 + "\n")

        columnas = [
            "Ticker",
            "Nombre",
            "Close",
            "Consensus_Probability",
            "Consensus_Score",
            "Score_Tecnico",
            "Score_Final",
            "Señal",
        ]

        columnas = [c for c in columnas if c in top.columns]

        for _, row in top[columnas].iterrows():
            f.write(
                f"{row.get('Ticker', ''):10} | "
                f"{row.get('Nombre', '')[:30]:30} | "
                f"Score: {row.get('Score_Final', 0):6.2f} | "
                f"Senal: {row.get('Señal', '')}\n"
            )

        f.write("\n")

        # ------------------------------------------------------------
        # PORTFOLIO RECOMENDADO
        # ------------------------------------------------------------
        f.write("-" * 80 + "\n")
        f.write("PORTFOLIO RECOMENDADO\n")
        f.write("-" * 80 + "\n")

        if portfolio.empty:
            f.write(
                "No hay portfolio recomendado: "
                "no se detectaron senales de compra.\n"
            )
        else:
            for _, row in portfolio.iterrows():
                f.write(
                    f"{row.get('Ticker', ''):10} | "
                    f"{row.get('Nombre', '')[:30]:30} | "
                    f"{row.get('Porcentaje', 0):6.2f}% | "
                    f"${row.get('Capital', 0):,.2f}\n"
                )

        f.write("\n")

        # ------------------------------------------------------------
        # INDICADORES TECNICOS DEL MERCADO
        # ------------------------------------------------------------
        f.write("-" * 80 + "\n")
        f.write("INDICADORES TECNICOS DEL MERCADO\n")
        f.write("-" * 80 + "\n")

        if indicadores.empty:
            f.write("No hay indicadores tecnicos agregados.\n")
        else:
            ind = indicadores.iloc[0]

            f.write(
                f"ADX (tendencia)              : "
                f"{formatear_indicador(ind.get('ADX_14'))}\n"
            )
            f.write(
                f"CMF (flujo dinero)           : "
                f"{formatear_indicador(ind.get('CMF_20'))}\n"
            )
            f.write(
                f"OBV (volumen acumulado)      : "
                f"{formatear_indicador(ind.get('OBV'))}\n"
            )
            f.write(
                f"VWAP                         : "
                f"{formatear_indicador(ind.get('VWAP'))}\n"
            )
            f.write(
                f"Volatilidad anualizada       : "
                f"{formatear_indicador(ind.get('Volatility_Annualized'), 4)}\n"
            )
            f.write(
                f"Stochastic %K                : "
                f"{formatear_indicador(ind.get('Stoch_%K'))}\n"
            )
            f.write(
                f"Stochastic %D                : "
                f"{formatear_indicador(ind.get('Stoch_%D'))}\n"
            )
            f.write(
                f"CCI                          : "
                f"{formatear_indicador(ind.get('CCI_20'))}\n"
            )
            f.write(
                f"Williams %R                  : "
                f"{formatear_indicador(ind.get('Williams_%R'))}\n"
            )
            f.write(
                f"Connors RSI (CRSI)           : "
                f"{formatear_indicador(ind.get('CRSI'))}\n"
            )

        f.write("\n")

        # ------------------------------------------------------------
        # CONCLUSION
        # ------------------------------------------------------------
        f.write("-" * 80 + "\n")
        f.write("CONCLUSION\n")
        f.write("-" * 80 + "\n")
        f.write(str(decision.get("Justificacion", "")))
        f.write("\n")

    print("=" * 80)
    print("DAILY REPORT GENERADO")
    print("=" * 80)
    print(output_txt)

    return output_txt


import sys

if __name__ == "__main__":
    mercado = sys.argv[1] if len(sys.argv) > 1 else "merval"
    generar_daily_report(mercado)
