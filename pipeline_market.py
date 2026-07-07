import sys
from pathlib import Path
from datetime import datetime
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from config import MarketConfig, SUPPORTED_MARKETS, MARKET_DEFINITIONS

from source.market_downloader import descargar_mercado
from source.market_indicators import generar_indicadores
from source.market_context_engine import generar_contexto_mercado
from source.market_dataset import generar_dataset_ia
from source.market_training import entrenar_mercado
from source.market_model_manager import actualizar_market_model_manager
from source.market_scoring import generar_market_scoring


def ejecutar_paso(numero, total, descripcion, funcion, mercado):
    print()
    print("=" * 80)
    print(f"[{numero}/{total}] {descripcion}")
    print("=" * 80)

    inicio = datetime.now()

    try:
        resultado = funcion(mercado)
        print()
        print(f"✅ {descripcion} finalizado correctamente")
        print(f"Tiempo paso: {datetime.now() - inicio}")
        return True, resultado

    except Exception as e:
        print()
        print(f"❌ Error en: {descripcion}")
        print("-" * 80)
        print(str(e))
        print()
        traceback.print_exc()
        return False, None


def ejecutar_pipeline(mercado="merval"):
    inicio_total = datetime.now()
    market = MarketConfig(mercado)

    print("=" * 80)
    print("                  MARKET AI PREDICTION PIPELINE")
    print("=" * 80)
    print(f"Mercado       : {market.display_name}")
    print(f"Descripción   : {market.description}")
    print(f"Activos       : {len(market.tickers)}")
    print(f"Target        : {market.target_days} días")
    print(f"Periodo       : {market.period}")
    print(f"Intervalo     : {market.interval}")
    print("=" * 80)

    pasos = [
        ("Descargar datos", descargar_mercado),
        ("Calcular indicadores", generar_indicadores),
        ("Generar contexto macro", generar_contexto_mercado),
        ("Generar dataset IA + contexto", generar_dataset_ia),
        ("Entrenar modelos", entrenar_mercado),
        ("Actualizar Model Manager", actualizar_market_model_manager),
        ("Generar scoring consensus", generar_market_scoring),
    ]

    resultados = []
    exitos = 0
    total = len(pasos)

    for i, (descripcion, funcion) in enumerate(pasos, start=1):
        ok, resultado = ejecutar_paso(i, total, descripcion, funcion, mercado)
        resultados.append((descripcion, ok))

        if ok:
            exitos += 1
        else:
            break

    print()
    print("=" * 80)
    print("RESUMEN PIPELINE")
    print("=" * 80)

    for descripcion, ok in resultados:
        estado = "OK" if ok else "ERROR"
        print(f"{estado:6} | {descripcion}")

    print("-" * 80)
    print(f"Mercado          : {market.display_name}")
    print(f"Pasos ejecutados : {len(resultados)}")
    print(f"Pasos OK         : {exitos}")
    print(f"Tiempo total     : {datetime.now() - inicio_total}")

    if exitos == total:
        print()
        print("✅ PIPELINE FINALIZADO CORRECTAMENTE")
    else:
        print()
        print("❌ PIPELINE INTERRUMPIDO POR ERROR")

    print("=" * 80)


def seleccionar_mercado():
    print()
    print("=" * 80)
    print("              MARKET AI PREDICTION PLATFORM")
    print("=" * 80)
    print("Seleccione el mercado a procesar:")
    print()

    mercados_disponibles = []

    for mercado in SUPPORTED_MARKETS:
        definicion = MARKET_DEFINITIONS.get(mercado, {})
        symbols = definicion.get("symbols", {})
        display = definicion.get("display_name", mercado.upper())

        if len(symbols) > 0:
            mercados_disponibles.append(mercado)

    for i, mercado in enumerate(mercados_disponibles, start=1):
        definicion = MARKET_DEFINITIONS[mercado]
        display = definicion["display_name"]
        descripcion = definicion["description"]
        activos = len(definicion["symbols"])

        print(f"{i:2}. {display:15} | {activos:3} activos | {descripcion}")

    print()
    print(" 0. Salir")
    print("=" * 80)

    opcion = input("Ingrese opción: ").strip()

    if opcion == "0":
        print("Saliendo...")
        sys.exit(0)

    try:
        idx = int(opcion) - 1

        if 0 <= idx < len(mercados_disponibles):
            mercado = mercados_disponibles[idx]
            print()
            print(f"Mercado seleccionado: {MARKET_DEFINITIONS[mercado]['display_name']}")
            return mercado

    except Exception:
        pass

    print()
    print("Opción inválida. Se usará MERVAL por defecto.")
    return "merval"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        mercado = sys.argv[1].lower().strip()
    else:
        mercado = seleccionar_mercado()

    ejecutar_pipeline(mercado)