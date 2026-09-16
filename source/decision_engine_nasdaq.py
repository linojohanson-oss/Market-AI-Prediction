import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reportes" / "nasdaq"

def generar_decision_nasdaq():
    ranking_path = REPORT_DIR / "ranking_diario.csv"
    output_path = REPORT_DIR / "decision_engine_nasdaq.csv"

    if not ranking_path.exists():
        raise FileNotFoundError(f"❌ Falta ranking diario: {ranking_path}")

    df = pd.read_csv(ranking_path)

    score_medio = df["Score_Final"].mean()
    compras = (df["Señal"] == "COMPRA").sum()

    market_ai_index = round(score_medio * 0.35, 2)

    if compras >= 3:
        semaforo = "🟢"
        estado = "POSITIVO"
        cash = 30
        acciones = 70
        max_pos = 5
    elif compras == 0:
        semaforo = "🔴"
        estado = "RIESGO ALTO"
        cash = 70
        acciones = 30
        max_pos = 2
    else:
        semaforo = "🟡"
        estado = "NEUTRAL"
        cash = 50
        acciones = 50
        max_pos = 3

    justificacion = (
        f"Score medio: {score_medio:.2f}. "
        f"Señales de compra: {compras}. "
        f"Market AI Index: {market_ai_index}. "
        f"Estado: {estado}."
    )

    data = {
        "Fecha": datetime.now().strftime("%Y-%m-%d"),
        "Decision": estado,
        "Market_AI_Index": market_ai_index,
        "Semaforo": semaforo,
        "Estado": estado,
        "Cash_pct": cash,
        "Acciones_pct": acciones,
        "Max_Posiciones": max_pos,
        "Justificacion": justificacion,
    }

    pd.DataFrame([data]).to_csv(output_path, index=False)
    print(f"✔ Decision NASDAQ generado: {output_path}")

if __name__ == "__main__":
    generar_decision_nasdaq()
