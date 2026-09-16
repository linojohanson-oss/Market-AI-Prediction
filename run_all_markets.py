import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from config import SUPPORTED_MARKETS, MARKET_DEFINITIONS


PROJECT_PATH = Path(__file__).resolve().parent
MARKET_PIPELINE = PROJECT_PATH / "pipeline_market_v2.py"
GLOBAL_INVESTMENT_REPORT = PROJECT_PATH / "source" / "global_investment_report.py"
GLOBAL_PDF_REPORT = PROJECT_PATH / "generar_pdf_con_graficos.py"

BASE_PATH = PROJECT_PATH / "reportes"
JSON_GLOBAL = BASE_PATH / "pipeline_maestro.json"


def obtener_mercados_activos():
    mercados = []
    for mercado in SUPPORTED_MARKETS:
        definicion = MARKET_DEFINITIONS.get(mercado, {})
        symbols = definicion.get("symbols", {})
        if len(symbols) > 0:
            mercados.append(mercado)
        else:
            print(
                f"AVISO {mercado.upper():15} omitido: "
                "no tiene activos configurados."
            )
    return mercados


def ejecutar_mercado(mercado):
    inicio = datetime.now()
    print()
    print("=" * 80)
    print(f"EJECUTANDO MERCADO: {mercado.upper()}")
    print("=" * 80)

    resultado = subprocess.run(
        [sys.executable, str(MARKET_PIPELINE), mercado],
        cwd=str(PROJECT_PATH),
    )
    tiempo = str(datetime.now() - inicio)
    estado = "OK" if resultado.returncode == 0 else "ERROR"
    return {
        "mercado": mercado,
        "estado": estado,
        "returncode": resultado.returncode,
        "tiempo": tiempo,
    }


def guardar_json_global(resultados):
    BASE_PATH.mkdir(parents=True, exist_ok=True)
    contenido = {
        "generado": datetime.now().isoformat(),
        "python": sys.executable,
        "pipeline": str(MARKET_PIPELINE),
        "resultados": resultados,
    }
    with open(JSON_GLOBAL, "w", encoding="utf-8") as archivo:
        json.dump(contenido, archivo, indent=4, ensure_ascii=False)
    print()
    print(f"OK Resumen global generado: {JSON_GLOBAL}")


def _ejecutar_etapa_final(nombre, archivo):
    print()
    print("=" * 80)
    print(nombre)
    print("=" * 80)

    if not archivo.exists():
        print(f"ERROR: no existe el programa requerido: {archivo}")
        return {"estado": "ERROR", "returncode": -1}

    resultado = subprocess.run(
        [sys.executable, str(archivo)],
        cwd=str(PROJECT_PATH),
    )
    estado = "OK" if resultado.returncode == 0 else "ERROR"
    if estado == "OK":
        print(f"OK {nombre}")
    else:
        print(f"ERROR {nombre} - codigo {resultado.returncode}")
    return {"estado": estado, "returncode": resultado.returncode}


def ejecutar_informe_global():
    return _ejecutar_etapa_final(
        "ETAPA FINAL 1/2 - GLOBAL INVESTMENT REPORT V3.1",
        GLOBAL_INVESTMENT_REPORT,
    )


def ejecutar_pdf_global():
    return _ejecutar_etapa_final(
        "ETAPA FINAL 2/2 - PDF GLOBAL CON GRAFICOS V2.1",
        GLOBAL_PDF_REPORT,
    )


def main():
    inicio_total = datetime.now()

    print()
    print("=" * 80)
    print("              MARKET AI PREDICTION - PIPELINE MAESTRO V2.1")
    print("=" * 80)
    print(f"Python   : {sys.executable}")
    print(f"Proyecto : {PROJECT_PATH}")
    print(f"Pipeline : {MARKET_PIPELINE}")
    print("=" * 80)

    if not MARKET_PIPELINE.exists():
        raise FileNotFoundError(f"No existe el pipeline principal:\n{MARKET_PIPELINE}")

    mercados = obtener_mercados_activos()
    print()
    print("Mercados que seran procesados:")
    for mercado in mercados:
        definicion = MARKET_DEFINITIONS[mercado]
        cantidad = len(definicion.get("symbols", {}))
        print(f"  - {mercado.upper():15} {cantidad:3} activos")

    resultados = []
    for mercado in mercados:
        resultados.append(ejecutar_mercado(mercado))

    guardar_json_global(resultados)

    print()
    print("=" * 80)
    print("RESUMEN PIPELINE MAESTRO V2.1")
    print("=" * 80)

    exitos = 0
    for resultado in resultados:
        if resultado["estado"] == "OK":
            exitos += 1
        print(
            f"{resultado['estado']:6} | "
            f"{resultado['mercado'].upper():15} | "
            f"{resultado['tiempo']}"
        )

    errores = len(resultados) - exitos
    print("-" * 80)
    print(f"Mercados ejecutados : {len(resultados)}")
    print(f"Mercados OK         : {exitos}")
    print(f"Mercados con error  : {errores}")
    print(f"Tiempo total        : {datetime.now() - inicio_total}")
    print("=" * 80)

    if errores != 0:
        print("AVISO PIPELINE MAESTRO FINALIZADO CON ERRORES")
        print(
            "No se generan el informe ni el PDF global para evitar "
            "conclusiones con mercados incompletos."
        )
        print("=" * 80)
        return False

    print("OK PIPELINE MAESTRO FINALIZADO CORRECTAMENTE")
    informe_global = ejecutar_informe_global()
    if informe_global["estado"] != "OK":
        print("AVISO: fallo el informe global V3.1; no se genera el PDF global.")
        print("=" * 80)
        return False

    pdf_global = ejecutar_pdf_global()
    print()
    print("=" * 80)
    if pdf_global["estado"] == "OK":
        print(
            "OK RUN ALL FINALIZADO CORRECTAMENTE - "
            f"{len(mercados)} MERCADOS + INFORME GLOBAL V3.1 + PDF GLOBAL V2.1"
        )
    else:
        print(
            "AVISO: los mercados y el informe global finalizaron, "
            "pero fallo el PDF global V2.1."
        )
    print("=" * 80)
    return pdf_global["estado"] == "OK"


if __name__ == "__main__":
    resultado_ok = main()
    sys.exit(0 if resultado_ok else 1)
