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
from source.market_backtester import ejecutar_backtest
from source.market_backtester_walk_forward import (
    ejecutar_backtest as ejecutar_walk_forward_backtest,
)

from source.market_regime_detector_v2 import generar_market_regime
from source.decision_engine import generar_decision
from source.portfolio_optimizer import construir_portafolio
from source.indicadores_resumen import generar_indicadores_resumen
from source.daily_report import generar_daily_report
from source.market_pdf_report import generar_reporte_pdf_market_ai
from source.market_macro_report import generar_informe_macro


def preparar_ranking_v2(mercado):
    print(
        "Ranking V2: no requiere conversion. "
        "Se usa ranking_diario.csv estandar."
    )
    return None


def generar_resumen_indicadores(mercado):
    """Genera el resumen tecnico universal para cualquier mercado."""
    return generar_indicadores_resumen(mercado)


def ejecutar_paso(numero, total, descripcion, funcion, mercado):
    print()
    print("=" * 80)
    print(f"[{numero}/{total}] {descripcion}")
    print("=" * 80)

    inicio = datetime.now()

    try:
        resultado = funcion(mercado)
        print()
        print(f"OK {descripcion} finalizado correctamente")
        print(f"Tiempo paso: {datetime.now() - inicio}")
        return True, resultado

    except Exception as exc:
        print()
        print(f"ERROR en: {descripcion}")
        print("-" * 80)
        print(str(exc))
        print()
        traceback.print_exc()
        return False, None


def ejecutar_pipeline(mercado="merval"):
    inicio_total = datetime.now()
    market = MarketConfig(mercado)
    contexto_ejecucion = {}

    def generar_scoring_y_conservar(nombre_mercado):
        scoring = generar_market_scoring(nombre_mercado)
        contexto_ejecucion["scoring"] = scoring
        return scoring

    def generar_pdf_con_scoring_actual(nombre_mercado):
        return generar_reporte_pdf_market_ai(
            nombre_mercado,
            scoring=contexto_ejecucion.get("scoring"),
        )

    print("=" * 80)
    print("              MARKET AI PREDICTION PIPELINE V2.1")
    print("=" * 80)
    print(f"Mercado       : {market.display_name}")
    print(f"Descripcion   : {market.description}")
    print(f"Activos       : {len(market.tickers)}")
    print(f"Target        : {market.target_days} dias")
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
        ("Generar scoring consensus", generar_scoring_y_conservar),
        ("Preparar ranking para V2", preparar_ranking_v2),
        ("Detectar regimen de mercado", generar_market_regime),
        ("Decision Engine", generar_decision),
        ("Portfolio Optimizer", construir_portafolio),
        ("Generar resumen indicadores", generar_resumen_indicadores),
        ("Market Backtester", ejecutar_backtest),
        ("Walk Forward Backtester", ejecutar_walk_forward_backtest),
        ("Daily Report", generar_daily_report),
        ("Reporte PDF", generar_pdf_con_scoring_actual),
        ("Informe institucional macro", generar_informe_macro),
    ]

    resultados = []
    exitos = 0
    total = len(pasos)

    for numero, (descripcion, funcion) in enumerate(pasos, start=1):
        ok, resultado = ejecutar_paso(
            numero,
            total,
            descripcion,
            funcion,
            mercado,
        )
        resultados.append((descripcion, ok))

        if ok:
            exitos += 1
        else:
            break

    print()
    print("=" * 80)
    print("RESUMEN PIPELINE V2.1")
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
        print("OK PIPELINE V2.1 FINALIZADO CORRECTAMENTE")
        pipeline_ok = True
    else:
        print()
        print("ERROR PIPELINE V2.1 INTERRUMPIDO POR ERROR")
        pipeline_ok = False

    print("=" * 80)
    return pipeline_ok


def seleccionar_mercado():
    print()
    print("=" * 80)
    print("              MARKET AI PREDICTION PLATFORM V2.1")
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

    for numero, mercado in enumerate(mercados_disponibles, start=1):
        definicion = MARKET_DEFINITIONS[mercado]
        display = definicion["display_name"]
        descripcion = definicion["description"]
        activos = len(definicion["symbols"])

        print(
            f"{numero:2}. {display:15} | "
            f"{activos:3} activos | {descripcion}"
        )

    print()
    print(" 0. Salir")
    print("=" * 80)

    opcion = input("Ingrese opcion: ").strip()

    if opcion == "0":
        print("Saliendo...")
        sys.exit(0)

    try:
        indice = int(opcion) - 1

        if 0 <= indice < len(mercados_disponibles):
            mercado = mercados_disponibles[indice]
            print()
            print(
                "Mercado seleccionado: "
                f"{MARKET_DEFINITIONS[mercado]['display_name']}"
            )
            return mercado

    except (TypeError, ValueError):
        pass

    print()
    print("Opcion invalida. Se usara MERVAL por defecto.")
    return "merval"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        mercado_seleccionado = sys.argv[1].lower().strip()
    else:
        mercado_seleccionado = seleccionar_mercado()

    resultado_ok = ejecutar_pipeline(mercado_seleccionado)
    sys.exit(0 if resultado_ok else 1)
