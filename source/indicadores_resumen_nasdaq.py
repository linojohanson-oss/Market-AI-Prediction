import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reportes" / "nasdaq"
DATA_DIR = ROOT / "datos" / "nasdaq" / "procesados"

def generar_indicadores_nasdaq():
    indicadores_path = DATA_DIR / "NASDAQ_INDICADORES.csv"
    output_path = REPORT_DIR / "indicadores_resumen_nasdaq.csv"

    if not indicadores_path.exists():
        pd.DataFrame().to_csv(output_path, index=False)
        print("⚠ No hay indicadores técnicos disponibles.")
        return

    df = pd.read_csv(indicadores_path)

    indicadores_posibles = [
        "ADX_14",
        "CMF_20",
        "OBV",
        "VWAP",
        "Volatility_Annualized",
        "Stoch_%K",
        "Stoch_%D",
        "CCI_20",
        "Williams_%R",
        "CRSI",
    ]

    resumen = {}

    for ind in indicadores_posibles:
        resumen[ind] = df[ind].mean() if ind in df.columns else None

    pd.DataFrame([resumen]).to_csv(output_path, index=False)
    print(f"✔ Indicadores NASDAQ generados: {output_path}")

if __name__ == "__main__":
    generar_indicadores_nasdaq()
