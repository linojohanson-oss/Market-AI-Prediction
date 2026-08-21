import os
from pathlib import Path
from datetime import datetime
import pandas as pd

from source.app_logger import log_print as print

ROOT = Path(__file__).resolve().parent.parent


def generar_daily_report(mercado="merval"):
    reportes_dir = ROOT / "reportes" / mercado

    ranking_path = reportes_dir / "ranking_diario.csv"
    regime_path = reportes_dir / f"market_regime_{mercado}_v2.csv"
    decision_path = reportes_dir / f"decision_engine_{mercado}.csv"
    portfolio_path = reportes_dir / f"portfolio_optimizer_{mercado}.csv"

    output_txt = reportes_dir / f"market_ai_daily_report_{mercado}.txt"

    if not ranking_path.exists():
        raise FileNotFoundError(f"No existe ranking: {ranking_path}")

    if not regime_path.exists():
        raise FileNotFoundError(f"No existe régimen: {regime_path}")

    if not decision_path.exists():
        raise FileNotFoundError(f"No existe decision engine: {decision_path}")

    ranking = pd.read_csv(ranking_path)
    regime = pd.read_csv(regime_path).iloc[0]
    decision = pd.read_csv(decision_path).iloc[0]

    portfolio = (
        pd.read_csv(portfolio_path)
        if portfolio_path.exists()
        else pd.DataFrame()
    )

    top = ranking.head(10)

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("MARKET AI DAILY REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Fecha                 : {regime.get('Fecha', '')}\n")
        f.write(f"Mercado               : {mercado.upper()}\n")
        f.write(f"Generado              : {datetime.now()}\n\n")

        f.write("-" * 80 + "\n")
        f.write("RÉGIMEN DE MERCADO\n")
        f.write("-" * 80 + "\n")
        f.write(f"Régimen               : {regime.get('Regimen', '')}\n")
        f.write(f"Acción recomendada    : {regime.get('Accion_Recomendada', '')}\n")
        f.write(f"Descripción           : {regime.get('Descripcion', '')}\n")
        f.write(f"Probabilidad media    : {regime.get('Prob_Media', 0):.4f}\n")
        f.write(f"Probabilidad máxima   : {regime.get('Prob_Max', 0):.4f}\n")
        f.write(f"Score medio           : {regime.get('Score_Medio', 0):.2f}\n\n")

        f.write("-" * 80 + "\n")
        f.write("DECISIÓN IA\n")
        f.write("-" * 80 + "\n")
        f.write(f"Decision              : {decision.get('Decision', '')}\n")
        f.write(f"Market AI Index       : {decision.get('Market_AI_Index', '')}\n")
        f.write(f"Semáforo              : {decision.get('Semaforo', '')} {decision.get('Estado', '')}\n")
        f.write(f"Cash recomendado      : {decision.get('Cash_pct', '')} %\n")
        f.write(f"Acciones              : {decision.get('Acciones_pct', '')} %\n")
        f.write(f"Máximo posiciones     : {decision.get('Max_Posiciones', '')}\n\n")

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
                f"Señal: {row.get('Señal', '')}\n"
            )

        f.write("\n")
        f.write("-" * 80 + "\n")
        f.write("PORTFOLIO RECOMENDADO\n")
        f.write("-" * 80 + "\n")

        if portfolio.empty:
            f.write("No hay portfolio generado.\n")
        else:
            for _, row in portfolio.iterrows():
                f.write(
                    f"{row.get('Ticker', ''):10} | "
                    f"{row.get('Nombre', '')[:30]:30} | "
                    f"{row.get('Porcentaje', 0):6.2f}% | "
                    f"${row.get('Capital', 0):,.2f}\n"
                )

        f.write("\n")
        f.write("-" * 80 + "\n")
        f.write("CONCLUSIÓN\n")
        f.write("-" * 80 + "\n")
        f.write(str(decision.get("Justificacion", "")))
        f.write("\n")

    print("=" * 80)
    print("DAILY REPORT GENERADO")
    print("=" * 80)
    print(output_txt)

    return output_txt


if __name__ == "__main__":
    generar_daily_report("merval")