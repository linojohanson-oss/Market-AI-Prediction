import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from pipeline_market import ejecutar_pipeline, seleccionar_mercado


DEPRECATION_MESSAGE = (
    "[DEPRECATED] pipeline_market_v2.py fue unificado en pipeline_market.py. "
    "Este wrapper se mantiene por compatibilidad y delega al pipeline principal."
)


def main():
    print(DEPRECATION_MESSAGE)

    if len(sys.argv) > 1:
        mercado = sys.argv[1].lower().strip()
    else:
        mercado = seleccionar_mercado()

    ok = ejecutar_pipeline(mercado)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
