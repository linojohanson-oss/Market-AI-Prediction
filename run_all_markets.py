import subprocess
import json
import os
from datetime import datetime
import sys
from pathlib import Path

from source.app_logger import configure_from_config, get_active_info_verbosity, log


def _configure_stdout_encoding():
    """Avoid UnicodeEncodeError on Windows terminals/codepages."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_configure_stdout_encoding()

PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_PATH = str(PROJECT_ROOT)
MARKET_PIPELINE = str(PROJECT_ROOT / "pipeline_market.py")
PYTHON_EXECUTABLE = str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")
ACCESS_VIOLATION_CODES = {3221225477, -1073741819}

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


def ejecutar_mercado(
    mercado,
    export_excel_indicators=False,
    export_excel_download=False,
    export_excel_dataset=False,
    export_excel_ranking=False,
    export_excel_model_ranking=False,
    export_excel_context=False,
    export_excel_all=False,
):
    inicio = datetime.now()

    log("\n" + "="*80, level="INFO", info_verbosity=1)
    log(f"[RUN] Ejecutando mercado: {mercado.upper()}", level="INFO", info_verbosity=1)
    log("="*80, level="INFO", info_verbosity=1)

    child_env = os.environ.copy()
    child_env["MARKET_EXPORT_EXCEL"] = "1" if export_excel_all else "0"
    child_env["MARKET_INDICATORS_EXPORT_EXCEL"] = "1" if export_excel_indicators else "0"
    child_env["MARKET_DOWNLOADER_EXPORT_EXCEL"] = "1" if export_excel_download else "0"
    child_env["MARKET_DATASET_EXPORT_EXCEL"] = "1" if export_excel_dataset else "0"
    child_env["MARKET_RANKING_EXPORT_EXCEL"] = "1" if export_excel_ranking else "0"
    child_env["MARKET_MODEL_RANKING_EXPORT_EXCEL"] = "1" if export_excel_model_ranking else "0"
    child_env["MARKET_CONTEXT_EXPORT_EXCEL"] = "1" if export_excel_context else "0"
    child_env["PYTHONUNBUFFERED"] = "1"
    child_env["PYTHONIOENCODING"] = "utf-8"

    comando = [PYTHON_EXECUTABLE, MARKET_PIPELINE, mercado]

    # Mostrar salida EN VIVO (sin capture_output)
    resultado = subprocess.run(
        comando,
        text=True,
        env=child_env,
        cwd=PROJECT_ROOT,
    )

    if resultado.returncode in ACCESS_VIOLATION_CODES:
        log(
            f"[WARN] Crash nativo detectado en {mercado.upper()} (exit={resultado.returncode}). Reintentando una vez...",
            level="WARNING",
        )
        resultado = subprocess.run(
            comando,
            text=True,
            env=child_env,
            cwd=PROJECT_ROOT,
        )

    estado = "OK" if resultado.returncode == 0 else "ERROR"
    tiempo = str(datetime.now() - inicio)

    return {
        "mercado": mercado,
        "estado": estado,
        "tiempo": tiempo,
        "codigo_salida": resultado.returncode
    }


def guardar_json_global(resultados):
    with open(JSON_GLOBAL, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=4)
    log(f"[OK] Archivo JSON global generado: {JSON_GLOBAL}", level="INFO", info_verbosity=2)


def main():
    resultados = []
    configured_level = configure_from_config()
    configured_info_verbosity = get_active_info_verbosity()

    log("\n" + "="*80, level="INFO", info_verbosity=1)
    log("            PIPELINE MAESTRO - MARKET AI PREDICTION", level="INFO", info_verbosity=1)
    log("="*80, level="INFO", info_verbosity=1)
    log(
        f"Log level configurado: {configured_level}"
        + (f" (info_verbosity={configured_info_verbosity})" if configured_level == "INFO" else ""),
        level="INFO",
        info_verbosity=1,
    )
    log(str(MARKETS), level="DEBUG")
    log(str(sys.argv), level="DEBUG")
    export_excel_indicators = False
    export_excel_download = False
    export_excel_dataset = False
    export_excel_ranking = False
    export_excel_model_ranking = False
    export_excel_context = False
    export_excel_all = False

    if len(sys.argv) > 1:
        args = sys.argv[1:]
        if "--excel-all" in args:
            export_excel_all = True
        if "--excel-indicators" in args:
            export_excel_indicators = True
        if "--excel-download" in args:
            export_excel_download = True
        if "--excel-dataset" in args:
            export_excel_dataset = True
        if "--excel-ranking-diario" in args:
            export_excel_ranking = True
        if "--excel-ranking-modelos" in args:
            export_excel_model_ranking = True
        if "--excel-contexto" in args:
            export_excel_context = True

        args = [
            a for a in args
            if a not in {
                "--excel-all",
                "--excel-indicators",
                "--excel-download",
                "--excel-dataset",
                "--excel-ranking-diario",
                "--excel-ranking-modelos",
                "--excel-contexto",
            }
        ]

        if export_excel_all:
            export_excel_indicators = True
            export_excel_download = True
            export_excel_dataset = True
            export_excel_ranking = True
            export_excel_model_ranking = True
            export_excel_context = True

        # Negative args (e.g. -crypto) always exclude those markets.
        excluded_markets = {arg[1:] for arg in args if arg.startswith("-")}
        filtered_markets = [x for x in MARKETS if x not in excluded_markets]

        # If there are positive args (e.g. crypto nasdaq), keep only those.
        included_markets = {arg for arg in args if not arg.startswith("-")}
        if included_markets:
            filtered_markets = [x for x in filtered_markets if x in included_markets]
    else:
        filtered_markets = MARKETS

    log(f"Mercados a ejecutar: {filtered_markets}", level="INFO", info_verbosity=2)
    log(
        f"Excel all          : {'ON' if export_excel_all else 'OFF'}",
        level="INFO",
        info_verbosity=2,
    )
    log(
        f"Excel indicadores  : {'ON' if export_excel_indicators else 'OFF'}",
        level="INFO",
        info_verbosity=2,
    )
    log(
        f"Excel descarga     : {'ON' if export_excel_download else 'OFF'}",
        level="INFO",
        info_verbosity=2,
    )
    log(
        f"Excel dataset      : {'ON' if export_excel_dataset else 'OFF'}",
        level="INFO",
        info_verbosity=2,
    )
    log(
        f"Excel ranking d.   : {'ON' if export_excel_ranking else 'OFF'}",
        level="INFO",
        info_verbosity=2,
    )
    log(
        f"Excel ranking mod.  : {'ON' if export_excel_model_ranking else 'OFF'}",
        level="INFO",
        info_verbosity=2,
    )
    log(
        f"Excel contexto     : {'ON' if export_excel_context else 'OFF'}",
        level="INFO",
        info_verbosity=2,
    )
    hubo_errores = False

    for mercado in filtered_markets:
        r = ejecutar_mercado(
            mercado,
            export_excel_indicators=export_excel_indicators,
            export_excel_download=export_excel_download,
            export_excel_dataset=export_excel_dataset,
            export_excel_ranking=export_excel_ranking,
            export_excel_model_ranking=export_excel_model_ranking,
            export_excel_context=export_excel_context,
            export_excel_all=export_excel_all,
        )
        resultados.append(r)
        if r["estado"] != "OK":
            log(f"Estado {mercado.upper():12}: {r['estado']} (exit={r['codigo_salida']})", level="ERROR")
        else:
            log(f"Estado {mercado.upper():12}: {r['estado']} (exit={r['codigo_salida']})", level="INFO", info_verbosity=2)
        if r["estado"] != "OK":
            hubo_errores = True

    guardar_json_global(resultados)

    if hubo_errores:
        log("\n[WARN] Se detectaron mercados con ERROR. Se omite generación de PDF global.", level="WARNING")
        log("[WARN] Revisá reportes/pipeline_maestro.json para ver el estado por mercado.", level="WARNING")
        log("\n" + "="*80, level="ERROR")
        log("[ERROR] Pipeline maestro finalizado con errores", level="ERROR")
        log("="*80, level="ERROR")
        sys.exit(1)

    log("\nGenerando reporte PDF con gráficos...", level="INFO", info_verbosity=2)
    pdf_result = subprocess.run([
        PYTHON_EXECUTABLE,
        os.path.join(PROJECT_PATH, "generar_pdf_con_graficos.py")
    ], text=True)
    if pdf_result.returncode != 0:
        log("\n" + "="*80, level="ERROR")
        log("[ERROR] Falló la generación del PDF global", level="ERROR")
        log("="*80, level="ERROR")
        sys.exit(pdf_result.returncode)

    log("\n" + "="*80, level="INFO", info_verbosity=1)
    log("[OK] Pipeline maestro finalizado correctamente", level="INFO", info_verbosity=1)
    log("="*80, level="INFO", info_verbosity=1)


if __name__ == "__main__":
    
    main()
