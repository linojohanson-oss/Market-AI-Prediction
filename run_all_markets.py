import subprocess
import json
import os
from datetime import datetime
import sys

PROJECT_PATH = r"."
MARKET_PIPELINE = os.path.join(PROJECT_PATH, "pipeline_market.py")

# MARKETS = [
#     "merval",
#     "crypto",
#     "nasdaq",
#     "etf",
#     "commodities",
#     "forex",
#     "bonos",
#     "indices"
# ]
MARKETS = [
    "merval",
    "crypto",
    "nasdaq",
    "etf",
    "bonos"
]
BASE_PATH = os.path.join(PROJECT_PATH, "reportes")
LOG_GLOBAL = os.path.join(BASE_PATH, "pipeline_maestro_salida.txt")
JSON_GLOBAL = os.path.join(BASE_PATH, "pipeline_maestro.json")


def ejecutar_mercado(mercado):
    inicio = datetime.now()

    print("\n" + "="*80)
    print(f"🚀 Ejecutando mercado: {mercado.upper()}")
    print("="*80)

    # Mostrar salida EN VIVO (sin capture_output)
    resultado = subprocess.run(
        ["python", MARKET_PIPELINE, mercado],
        text=True
    )

    estado = "OK" if resultado.returncode == 0 else "ERROR"
    tiempo = str(datetime.now() - inicio)

    return {
        "mercado": mercado,
        "estado": estado,
        "tiempo": tiempo
    }


def guardar_json_global(resultados):
    with open(JSON_GLOBAL, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=4)
    print("✔ Archivo JSON global generado:", JSON_GLOBAL)


def main():
    resultados = []

    print("\n" + "="*80)
    print("            PIPELINE MAESTRO - MARKET AI PREDICTION")
    print("="*80)
    filtered_markets = [x for x in MARKETS if '-'+x not in sys.argv]
    filtered_markets = [x for x in filtered_markets if x  in sys.argv]

    for mercado in MARKETS:
        r = ejecutar_mercado(mercado)
        resultados.append(r)
        

    guardar_json_global(resultados)
    print("\nGenerando reporte PDF con gráficos...")
    subprocess.run(["python", os.path.join(PROJECT_PATH, "generar_pdf_con_graficos.py")])

    print("\n" + "="*80)
    print("✔ Pipeline maestro finalizado correctamente")
    print("="*80)


if __name__ == "__main__":
    
    main()
