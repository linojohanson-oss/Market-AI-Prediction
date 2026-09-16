import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = r"C:\Users\feder\AppData\Local\Programs\Python\Python39\python.exe"

def run(args):
    print("\n" + "="*80)
    print(f">>> Ejecutando: {' '.join(args)}")
    print("="*80)
    subprocess.run(args, check=True)

if __name__ == "__main__":
    print("="*80)
    print("               EJECUTANDO PIPELINE DIARIO NASDAQ")
    print("="*80)

    run([PYTHON, str(ROOT / "market_regime_nasdaq_v2.py")])
    run([PYTHON, str(ROOT / "decision_engine_nasdaq.py")])
    run([PYTHON, str(ROOT / "portfolio_optimizer_nasdaq.py")])
    run([PYTHON, str(ROOT / "indicadores_resumen_nasdaq.py")])

    # 🔥 Parámetro NASDAQ ahora se pasa correctamente
    run([PYTHON, str(ROOT / "daily_report.py"), "nasdaq"])

    print("="*80)
    print("✔ DAILY REPORT NASDAQ GENERADO EXITOSAMENTE")
    print("="*80)
