import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

def run(cmd):
    print(f"\n>>> Ejecutando: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    print("=" * 80)
    print("EJECUTANDO PIPELINE DIARIO MERVAL")
    print("=" * 80)

    run(f'"{PYTHON}" "{ROOT}/market_regime_merval_v2.py"')
    run(f'"{PYTHON}" "{ROOT}/decision_engine_merval.py"')
    run(f'"{PYTHON}" "{ROOT}/portfolio_optimizer_merval.py"')
    run(f'"{PYTHON}" "{ROOT}/indicadores_resumen_merval.py"')
    run(f'"{PYTHON}" "{ROOT}/daily_report.py" merval')

    print("\n" + "=" * 80)
    print("✔ DAILY REPORT MERVAL GENERADO EXITOSAMENTE")
    print("=" * 80)
