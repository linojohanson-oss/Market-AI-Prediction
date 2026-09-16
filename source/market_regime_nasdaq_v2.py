import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reportes" / "nasdaq"

def generar_regimen_nasdaq():
    ranking_path = REPORT_DIR / "ranking_diario.csv"
    output_path = REPORT_DIR / "market_regime_nasdaq_v2.csv"

    if not ranking_path.exists():
        raise FileNotFoundError(f"❌ Falta ranking diario: {ranking_path}")

    df = pd.read_csv(ranking_path)

    score_medio = df["Score_Final"].mean()
    prob_media = df["Consensus_Probability"].mean()
    prob_max = df["Consensus_Probability"].max()

    if score_medio >= 70:
        regimen = "ALCISTA"
        accion = "COMPRAR"
        descripcion = "Fuerza positiva en el mercado."
    elif score_medio <= 45:
        regimen = "BAJISTA"
        accion = "EVITAR"
        descripcion = "Mercado débil."
    else:
        regimen = "NEUTRAL"
        accion = "VIGILAR"
        descripcion = "Mercado sin tendencia clara."

    data = {
        "Fecha": datetime.now().strftime("%Y-%m-%d"),
        "Regimen": regimen,
        "Accion_Recomendada": accion,
        "Descripcion": descripcion,
        "Prob_Media": prob_media,
        "Prob_Max": prob_max,
        "Score_Medio": score_medio,
    }

    pd.DataFrame([data]).to_csv(output_path, index=False)
    print(f"✔ Régimen NASDAQ generado: {output_path}")

if __name__ == "__main__":
    generar_regimen_nasdaq()
