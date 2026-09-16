import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reportes" / "crypto"
DATA_DIR = ROOT / "datos" / "crypto" / "procesados"

def generar_indicadores_crypto():
    indicadores_path = DATA_DIR / "CRYPTO_INDICADORES.csv"
    output_path = REPORT_DIR / "indicadores_resumen_crypto.csv"

    if not indicadores_path.exists():
        pd.DataFrame().to_csv(output_path, index=False)
        print("⚠ No hay indicadores técnicos disponibles.")
        return

    df = pd.read_csv(indicadores_path)

    resumen = {
        "ADX_14": df["ADX_14"].mean(),
        "CMF_20": df["CMF_20"].mean(),
        "OBV": df["OBV"].mean(),
        "VWAP": df["VWAP"].mean(),
        "Volatility_Annualized": df["Volatility_Annualized"].mean(),
        "Stoch_%K": df["Stoch_%K"].mean(),
        "Stoch_%D": df["Stoch_%D"].mean(),
        "CCI_20": df["CCI_20"].mean(),
        "Williams_%R": df["Williams_%R"].mean(),
        "CRSI": df["CRSI"].mean(),
    }

    pd.DataFrame([resumen]).to_csv(output_path, index=False)
    print(f"✔ Indicadores técnicos generados: {output_path}")

if __name__ == "__main__":
    generar_indicadores_crypto()
