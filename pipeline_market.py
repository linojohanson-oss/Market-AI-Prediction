import sys
import os
from pathlib import Path
from datetime import datetime
import traceback
import shutil
import io
from contextlib import redirect_stdout, redirect_stderr

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))


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

from config import MarketConfig, SUPPORTED_MARKETS, MARKET_DEFINITIONS, get_json_config_value

from source.market_downloader import descargar_mercado
from source.market_indicators import generar_indicadores
from source.market_context_engine import generar_contexto_mercado
from source.market_dataset import generar_dataset_ia
from source.market_training import entrenar_mercado
from source.market_model_manager import actualizar_market_model_manager
from source.market_scoring import generar_market_scoring
from source.market_regime_detector_v2 import generar_market_regime
from source.decision_engine import generar_decision
from source.portfolio_optimizer import construir_portafolio
from source.daily_report import generar_daily_report
from source.app_logger import (
    configure_from_config,
    get_active_level_name,
    get_active_info_verbosity,
    log,
    emit_captured_output,
)


def publicar_ranking_estable(nombre_mercado="merval"):
    """Publica el ranking timestamped a nombre estable para consumidores V2."""
    market = MarketConfig(nombre_mercado)
    reportes_dir = ROOT / "reportes" / market.market

    ranking_csv_files = sorted(reportes_dir.glob("ranking_diario_*.csv"))
    ranking_xlsx_files = sorted(reportes_dir.glob("ranking_diario_*.xlsx"))

    src_csv = ranking_csv_files[-1] if ranking_csv_files else None
    src_xlsx = ranking_xlsx_files[-1] if ranking_xlsx_files else None
    dst_csv = reportes_dir / "ranking_diario.csv"
    dst_xlsx = reportes_dir / "ranking_diario.xlsx"

    if src_csv is None:
        raise FileNotFoundError(
            f"No existe ranking timestamped en {reportes_dir} (pattern: ranking_diario_*.csv)"
        )

    shutil.copy2(src_csv, dst_csv)
    log(f"[SYNC] Ranking estable CSV: {dst_csv}", level="INFO", info_verbosity=3)

    if src_xlsx is not None:
        shutil.copy2(src_xlsx, dst_xlsx)
        log(f"[SYNC] Ranking estable Excel: {dst_xlsx}", level="INFO", info_verbosity=3)
    else:
        pd.read_csv(src_csv).to_excel(dst_xlsx, index=False)
        log(
            f"[SYNC] Ranking estable Excel reconstruido desde CSV: {dst_xlsx}",
            level="INFO",
            info_verbosity=3,
        )

    return {
        "ranking_csv": str(dst_csv),
        "ranking_xlsx": str(dst_xlsx),
    }


def ejecutar_con_log_filtrado(funcion, mercado):
    level = get_active_level_name()
    info_verbosity = get_active_info_verbosity()

    if level == "DEBUG":
        return funcion(mercado)

    if level == "INFO" and info_verbosity >= 5:
        return funcion(mercado)

    buffer = io.StringIO()

    with redirect_stdout(buffer), redirect_stderr(buffer):
        resultado = funcion(mercado)

    emit_captured_output(buffer.getvalue())
    return resultado


def ejecutar_paso(numero, total, descripcion, funcion, mercado):
    log("", level="INFO", info_verbosity=1)
    log("=" * 80, level="INFO", info_verbosity=1)
    log(f"[{numero}/{total}] {descripcion}", level="INFO", info_verbosity=1)
    log("=" * 80, level="INFO", info_verbosity=1)

    inicio = datetime.now()

    try:
        resultado = ejecutar_con_log_filtrado(funcion, mercado)
        log("", level="INFO", info_verbosity=2)
        log(f"[OK] {descripcion} finalizado correctamente", level="INFO", info_verbosity=2)
        log(f"Tiempo paso: {datetime.now() - inicio}", level="INFO", info_verbosity=2)
        return True, resultado

    except Exception as e:
        log("", level="ERROR")
        log(f"[ERROR] Error en: {descripcion}", level="ERROR")
        log("-" * 80, level="ERROR")
        log(str(e), level="ERROR")
        log("", level="ERROR")
        log(traceback.format_exc(), level="ERROR")
        return False, None


def ejecutar_pipeline(mercado="merval"):
    inicio_total = datetime.now()
    market = MarketConfig(mercado)
    configured_level = configure_from_config()
    configured_info_verbosity = get_active_info_verbosity()

    log("=" * 80, level="INFO", info_verbosity=1)
    log("                  MARKET AI PREDICTION PIPELINE", level="INFO", info_verbosity=1)
    log("=" * 80, level="INFO", info_verbosity=1)
    log(f"Mercado       : {market.display_name}", level="INFO", info_verbosity=1)
    log(f"Descripción   : {market.description}", level="INFO", info_verbosity=2)
    log(f"Activos       : {len(market.tickers)}", level="INFO", info_verbosity=2)
    log(f"Target        : {market.target_days} días", level="INFO", info_verbosity=2)
    target_days = int(market.target_days)
    target_min_return = get_json_config_value(
        "pipeline.target_min_return",
        get_json_config_value(f"pipeline.target_min_return_{target_days}d", 0.0),
    )
    try:
        target_min_return = float(target_min_return)
    except Exception:
        target_min_return = 0.0

    stop_loss_level = get_json_config_value(
        f"portfolio.markets.{market.market}.stop_loss_pct",
        get_json_config_value("portfolio.stop_loss_pct", 0.0),
    )
    try:
        stop_loss_level = float(stop_loss_level)
    except Exception:
        stop_loss_level = 0.0

    log(f"Target min %+ : {target_min_return * 100:.2f}%", level="INFO", info_verbosity=2)
    log(f"Stop loss     : {stop_loss_level * 100:.2f}%", level="INFO", info_verbosity=2)
    if market.period_was_adjusted:
        log(f"Periodo cfg   : {market.configured_period}", level="INFO", info_verbosity=2)
        log(f"Periodo usado : {market.effective_period}", level="INFO", info_verbosity=2)
        log(market.period_adjustment_message, level="WARNING")
    else:
        log(f"Periodo       : {market.period}", level="INFO", info_verbosity=2)
    log(f"Intervalo     : {market.interval}", level="INFO", info_verbosity=2)
    enable_optimization = bool(get_json_config_value("pipeline.enable_optimization", False))
    publish_stable_ranking = bool(get_json_config_value("pipeline.publish_stable_ranking", True))
    log(
        f"Log level     : {configured_level}"
        + (f" (info_verbosity={configured_info_verbosity})" if configured_level == "INFO" else ""),
        level="INFO",
        info_verbosity=1,
    )
    log(f"Optimización  : {'ON' if enable_optimization else 'OFF'}", level="INFO", info_verbosity=2)
    log(f"Sync ranking  : {'ON' if publish_stable_ranking else 'OFF'}", level="INFO", info_verbosity=2)
    skip_training = os.getenv("MARKET_SKIP_TRAINING", "0") == "1"
    log(f"Skip training : {'ON' if skip_training else 'OFF'}", level="INFO", info_verbosity=2)
    log("=" * 80, level="INFO", info_verbosity=1)

    pasos = [
        ("Descargar datos", descargar_mercado),
        ("Calcular indicadores", generar_indicadores),
        ("Generar contexto macro", generar_contexto_mercado),
        ("Generar dataset IA + contexto", generar_dataset_ia),
        ("Generar scoring consensus", generar_market_scoring),
    ]

    if not skip_training:
        pasos.insert(4, ("Entrenar modelos", entrenar_mercado))

    pasos.insert(5 if not skip_training else 4, ("Actualizar Model Manager", actualizar_market_model_manager))

    if publish_stable_ranking:
        pasos.append(("Publicar ranking estable (sin timestamp)", publicar_ranking_estable))

    if enable_optimization:
        pasos.extend([
            ("Detectar régimen de mercado", generar_market_regime),
            ("Decision Engine", generar_decision),
            ("Portfolio Optimizer", construir_portafolio),
            ("Daily Report", generar_daily_report),
        ])

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

    log("", level="INFO", info_verbosity=1)
    log("=" * 80, level="INFO", info_verbosity=1)
    log("RESUMEN PIPELINE", level="INFO", info_verbosity=1)
    log("=" * 80, level="INFO", info_verbosity=1)

    for descripcion, ok in resultados:
        estado = "OK" if ok else "ERROR"
        if estado == "ERROR":
            log(f"{estado:6} | {descripcion}", level="ERROR")
        else:
            log(f"{estado:6} | {descripcion}", level="INFO", info_verbosity=2)

    log("-" * 80, level="INFO", info_verbosity=1)
    log(f"Mercado          : {market.display_name}", level="INFO", info_verbosity=1)
    log(f"Pasos ejecutados : {len(resultados)}", level="INFO", info_verbosity=1)
    log(f"Pasos OK         : {exitos}", level="INFO", info_verbosity=1)
    log(f"Tiempo total     : {datetime.now() - inicio_total}", level="INFO", info_verbosity=2)

    if exitos == total:
        log("", level="INFO", info_verbosity=1)
        log("[OK] PIPELINE FINALIZADO CORRECTAMENTE", level="INFO", info_verbosity=1)
    else:
        log("", level="ERROR")
        log("[ERROR] PIPELINE INTERRUMPIDO POR ERROR", level="ERROR")

    log("=" * 80, level="INFO", info_verbosity=1)

    return exitos == total


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
    args = sys.argv[1:]
    enable_skip_training = "--skip-training" in args
    disable_skip_training = "--no-skip-training" in args
    enable_live_prices = "--live" in args
    disable_live_prices = "--no-live" in args
    enable_excel_all = "--excel-all" in args
    disable_excel_all = "--no-excel-all" in args
    enable_excel = "--excel-indicators" in args
    disable_excel = "--no-excel-indicators" in args
    enable_download_excel = "--excel-download" in args
    disable_download_excel = "--no-excel-download" in args
    enable_dataset_excel = "--excel-dataset" in args
    disable_dataset_excel = "--no-excel-dataset" in args
    enable_ranking_excel = "--excel-ranking-diario" in args
    disable_ranking_excel = "--no-excel-ranking-diario" in args
    enable_model_ranking_excel = "--excel-ranking-modelos" in args
    disable_model_ranking_excel = "--no-excel-ranking-modelos" in args
    enable_context_excel = "--excel-contexto" in args
    disable_context_excel = "--no-excel-contexto" in args

    args_clean = [
        a for a in args
        if a not in {
            "--skip-training",
            "--no-skip-training",
            "--live",
            "--no-live",
            "--excel-indicators",
            "--no-excel-indicators",
            "--excel-download",
            "--no-excel-download",
            "--excel-dataset",
            "--no-excel-dataset",
            "--excel-ranking-diario",
            "--no-excel-ranking-diario",
            "--excel-ranking-modelos",
            "--no-excel-ranking-modelos",
            "--excel-contexto",
            "--no-excel-contexto",
            "--excel-all",
            "--no-excel-all",
        }
    ]

    if disable_excel_all:
        os.environ["MARKET_EXPORT_EXCEL"] = "0"
    elif enable_excel_all:
        os.environ["MARKET_EXPORT_EXCEL"] = "1"
    else:
        os.environ.setdefault("MARKET_EXPORT_EXCEL", "0")

    global_excel_default = "1" if os.environ.get("MARKET_EXPORT_EXCEL", "0") == "1" else "0"

    if disable_excel:
        os.environ["MARKET_INDICATORS_EXPORT_EXCEL"] = "0"
    elif enable_excel:
        os.environ["MARKET_INDICATORS_EXPORT_EXCEL"] = "1"
    else:
        os.environ.setdefault("MARKET_INDICATORS_EXPORT_EXCEL", global_excel_default)

    if disable_download_excel:
        os.environ["MARKET_DOWNLOADER_EXPORT_EXCEL"] = "0"
    elif enable_download_excel:
        os.environ["MARKET_DOWNLOADER_EXPORT_EXCEL"] = "1"
    else:
        os.environ.setdefault("MARKET_DOWNLOADER_EXPORT_EXCEL", global_excel_default)

    if disable_dataset_excel:
        os.environ["MARKET_DATASET_EXPORT_EXCEL"] = "0"
    elif enable_dataset_excel:
        os.environ["MARKET_DATASET_EXPORT_EXCEL"] = "1"
    else:
        os.environ.setdefault("MARKET_DATASET_EXPORT_EXCEL", global_excel_default)

    if disable_ranking_excel:
        os.environ["MARKET_RANKING_EXPORT_EXCEL"] = "0"
    elif enable_ranking_excel:
        os.environ["MARKET_RANKING_EXPORT_EXCEL"] = "1"
    else:
        os.environ.setdefault("MARKET_RANKING_EXPORT_EXCEL", global_excel_default)

    if disable_model_ranking_excel:
        os.environ["MARKET_MODEL_RANKING_EXPORT_EXCEL"] = "0"
    elif enable_model_ranking_excel:
        os.environ["MARKET_MODEL_RANKING_EXPORT_EXCEL"] = "1"
    else:
        os.environ.setdefault("MARKET_MODEL_RANKING_EXPORT_EXCEL", global_excel_default)

    if disable_context_excel:
        os.environ["MARKET_CONTEXT_EXPORT_EXCEL"] = "0"
    elif enable_context_excel:
        os.environ["MARKET_CONTEXT_EXPORT_EXCEL"] = "1"
    else:
        os.environ.setdefault("MARKET_CONTEXT_EXPORT_EXCEL", global_excel_default)

    if disable_skip_training:
        os.environ["MARKET_SKIP_TRAINING"] = "0"
    elif enable_skip_training:
        os.environ["MARKET_SKIP_TRAINING"] = "1"
    else:
        os.environ.setdefault("MARKET_SKIP_TRAINING", "0")

    if disable_live_prices:
        os.environ["MARKET_LIVE_PRICES"] = "0"
    elif enable_live_prices:
        os.environ["MARKET_LIVE_PRICES"] = "1"
    else:
        os.environ.setdefault("MARKET_LIVE_PRICES", "0")

    if args_clean:
        mercado = args_clean[0].lower().strip()
    else:
        mercado = seleccionar_mercado()

    ok = ejecutar_pipeline(mercado)
    sys.exit(0 if ok else 1)