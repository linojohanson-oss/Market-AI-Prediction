import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reportes" / "crypto"

def generar_portfolio_crypto(capital_total=40000):
    ranking_path = REPORT_DIR / "ranking_diario.csv"
    output_path = REPORT_DIR / "portfolio_optimizer_crypto.csv"

    if not ranking_path.exists():
        raise FileNotFoundError(f"❌ Falta ranking diario: {ranking_path}")

    df = pd.read_csv(ranking_path)

    compras = df[df["Señal"] == "COMPRA"].copy()

    if compras.empty:
        pd.DataFrame().to_csv(output_path, index=False)
        print("⚠ No hay señales de compra. Portfolio vacío.")
        return

    compras["Peso"] = compras["Score_Final"] / compras["Score_Final"].sum()
    compras["Capital"] = compras["Peso"] * capital_total
    compras["Porcentaje"] = compras["Peso"] * 100

    compras.to_csv(output_path, index=False)
    print(f"✔ Portfolio generado: {output_path}")

if __name__ == "__main__":
    generar_portfolio_crypto()
