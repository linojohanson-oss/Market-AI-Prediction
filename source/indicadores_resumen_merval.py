import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reportes" / "merval"
DATA_DIR = ROOT / "datos" / "merval" / "procesados"

def generar_indicadores_merval():
    indicadores_path = DATA_DIR / "MERVAL_INDICADORES.csv"
    output_path = REPORT_DIR / "indicadores_resumen_merval.csv"

    if not indicadores_path.exists():
        pd.DataFrame().to_csv(output_path, index=False)
        print("⚠ No hay indicadores técnicos disponibles.")
        return

    df = pd.read_csv(indicadores_path)

    # Lista de indicadores posibles
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

    # Solo calcular los indicadores que existan en el CSV
    for ind in indicadores_posibles:
        if ind in df.columns:
            resumen[ind] = df[ind].mean()
        else:
            resumen[ind] = None  # o 0 si preferís

    pd.DataFrame([resumen]).to_csv(output_path, index=False)
    print(f"✔ Indicadores MERVAL generados: {output_path}")

if __name__ == "__main__":
    generar_indicadores_merval()
