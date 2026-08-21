import os
import pandas as pd
import json

from source.app_logger import log_print as print

BASE_PATH = r"C:\Users\feder\OneDrive\Desktop\OneDrive\PycharmProjects\Market-AI-Prediction\reportes"

MERCADOS = {
    "merval": "MERVAL",
    "crypto": "CRYPTO",
    "nasdaq": "NASDAQ",
    "etf": "ETF",
    "forex": "FOREX",
    "commodities": "COMMODITIES",
    "bonos": "BONOS",
    "indices": "INDICES"
}

def cargar_ranking(mercado):
    ruta = os.path.join(BASE_PATH, mercado, "ranking_diario.csv")
    if not os.path.exists(ruta):
        return None
    return pd.read_csv(ruta)

def cargar_modelo_activo(mercado):
    ruta = os.path.join(BASE_PATH, mercado, "modelo_activo.json")
    if not os.path.exists(ruta):
        return None
    with open(ruta, "r") as f:
        return json.load(f)

def procesar_mercado(mercado_key):
    nombre = MERCADOS[mercado_key]
    ranking = cargar_ranking(mercado_key)
    modelo = cargar_modelo_activo(mercado_key)

    if ranking is None:
        return {
            "mercado": nombre,
            "status": "sin datos"
        }

    señales = {
        "COMPRA_FUERTE": [],
        "COMPRA": [],
        "VIGILAR": [],
        "NEUTRAL": [],
        "EVITAR": []
    }

    for _, fila in ranking.iterrows():
        señal = fila["Señal"].strip().upper()
        ticker = fila["Ticker"]
        score = fila["Score_Final"]

        if señal in señales:
            señales[señal].append({
                "ticker": ticker,
                "score": score
            })

    return {
        "mercado": nombre,
        "modelo_activo": modelo,
        "señales": señales
    }

def generar_resumen_global():
    resumen = {}

    print("\n" + "="*80)
    print("               RESUMEN GLOBAL DE MERCADOS")
    print("="*80)

    for mercado_key in MERCADOS:
        info = procesar_mercado(mercado_key)
        resumen[mercado_key] = info

        print(f"\n--- {info['mercado']} ---")

        if info.get("status") == "sin datos":
            print("Sin datos.")
            continue

        modelo = info["modelo_activo"]
        print(f"Modelo activo: {modelo['modelo']} | Score: {modelo['score_modelo']}")

        señales = info["señales"]

        for tipo, lista in señales.items():
            if lista:
                print(f"  {tipo}:")
                for s in lista:
                    print(f"     - {s['ticker']} (Score {s['score']:.2f})")

    # Guardar JSON global
    ruta_json = os.path.join(BASE_PATH, "resumen_global.json")
    with open(ruta_json, "w") as f:
        json.dump(resumen, f, indent=4)

    print("\n" + "="*80)
    print("✔ Resumen global generado correctamente")
    print(f"Archivo: {ruta_json}")
    print("="*80)

if __name__ == "__main__":
    generar_resumen_global()
