import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = r"C:\Users\feder\AppData\Local\Programs\Python\Python39\python.exe"

def run(cmd):
    print(f"\n>>> Ejecutando: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    print("=" * 80)
    print("EJECUTANDO PIPELINE DIARIO CRYPTO")
    print("=" * 80)

    # 1. Régimen
    run(f'"{PYTHON}" "{ROOT}/market_regime_crypto_v2.py"')

    # 2. Decision Engine
    run(f'"{PYTHON}" "{ROOT}/decision_engine_crypto.py"')

    # 3. Portfolio Optimizer
    run(f'"{PYTHON}" "{ROOT}/portfolio_optimizer_crypto.py"')

    # 4. Indicadores Técnicos
    run(f'"{PYTHON}" "{ROOT}/indicadores_resumen_crypto.py"')

    # 5. Daily Report
    run(f'"{PYTHON}" "{ROOT}/daily_report.py" crypto')

    print("\n" + "=" * 80)
    print("✔ DAILY REPORT CRYPTO GENERADO EXITOSAMENTE")
    print("=" * 80)
