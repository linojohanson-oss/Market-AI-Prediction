# portfolio_optimizer.py

import os
from pathlib import Path
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent

from source.app_logger import log_print as print

from config import MarketConfig, get_json_config_value


CAPITAL_INICIAL = 100000.0

MAX_PESO_ACTIVO = 45.0
MIN_PESO_ACTIVO = 8.0
TOP_CANDIDATOS_EXTRA = 8


def es_mercado_crypto(nombre_mercado):
    return str(nombre_mercado).strip().lower() == "crypto"


def ticker_para_lookup(ticker, nombre_mercado):
    t = str(ticker or "").strip().upper().replace("/", "-")

    if not t:
        return t

    if es_mercado_crypto(nombre_mercado):
        if t.endswith("-USDT"):
            return f"{t[:-5]}-USD"
        if t.endswith("USDT") and "-" not in t:
            return f"{t[:-4]}-USD"

    return t


def ticker_para_output(ticker, nombre_mercado):
    t = str(ticker or "").strip().upper()

    if not t:
        return t

    if es_mercado_crypto(nombre_mercado) and t.endswith("-USD"):
        return f"{t[:-4]}-USDT"

    return t


def _parse_bool_env(value, default=False):
    if value is None:
        return default

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def exportar_excel_portfolio_habilitado():
    global_default = _parse_bool_env(os.getenv("MARKET_EXPORT_EXCEL"), default=False)
    return _parse_bool_env(os.getenv("MARKET_PORTFOLIO_EXPORT_EXCEL"), default=global_default)


def usar_precios_en_vivo():
    return _parse_bool_env(os.getenv("MARKET_LIVE_PRICES"), default=False)


def _obtener_timestamp_reportes(reportes_dir):
    candidatos = sorted(reportes_dir.glob("ranking_diario_*.csv"))
    if not candidatos:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    ultimo = candidatos[-1].stem
    prefijo = "ranking_diario_"
    if ultimo.startswith(prefijo):
        return ultimo[len(prefijo):]

    return datetime.now().strftime("%Y%m%d_%H%M%S")


@lru_cache(maxsize=512)
def _precio_en_vivo_yfinance(ticker):
    ticker = str(ticker or "").strip()
    if not ticker:
        return None

    try:
        instrumento = yf.Ticker(ticker)
        fast_info = getattr(instrumento, "fast_info", None)
        if fast_info is not None:
            for clave in ("lastPrice", "last_price", "regularMarketPrice", "previousClose"):
                try:
                    valor = fast_info.get(clave) if hasattr(fast_info, "get") else getattr(fast_info, clave, None)
                except Exception:
                    valor = None
                if valor is not None and pd.notna(valor):
                    valor = float(valor)
                    if valor > 0:
                        return valor
    except Exception:
        pass

    try:
        info = yf.Ticker(ticker).info or {}
        for clave in ("regularMarketPrice", "currentPrice", "previousClose"):
            valor = info.get(clave)
            if valor is not None and pd.notna(valor):
                valor = float(valor)
                if valor > 0:
                    return valor
    except Exception:
        pass

    try:
        df = yf.download(
            ticker,
            period="1d",
            interval="1m",
            progress=False,
            auto_adjust=False,
            threads=False,
        )
        if df is not None and not df.empty and "Close" in df.columns:
            cierre = pd.to_numeric(df["Close"], errors="coerce").dropna()
            if not cierre.empty:
                valor = float(cierre.iloc[-1])
                if valor > 0:
                    return valor
    except Exception:
        pass

    return None


def rutas_portfolio(nombre_mercado):
    market = MarketConfig(nombre_mercado)
    reportes_dir = ROOT / "reportes" / market.market
    timestamp = _obtener_timestamp_reportes(reportes_dir)

    return {
        "market": market,
        "ranking": reportes_dir / "ranking_diario.csv",
        "decision": reportes_dir / f"decision_engine_{market.market}.csv",
        "csv_ts_out": reportes_dir / f"portfolio_optimizer_{market.market}_{timestamp}.csv",
        "csv_out": reportes_dir / f"portfolio_optimizer_{market.market}.csv",
        "xlsx_out": reportes_dir / f"portfolio_optimizer_{market.market}.xlsx",
        "txt_out": reportes_dir / f"portfolio_optimizer_{market.market}.txt",
    }


def verificar_archivos(rutas):
    if not rutas["ranking"].exists():
        raise FileNotFoundError(f"No existe ranking diario: {rutas['ranking']}")

    if not rutas["decision"].exists():
        raise FileNotFoundError(f"No existe Decision Engine: {rutas['decision']}")


def cargar_decision(rutas):
    return pd.read_csv(rutas["decision"]).iloc[0]


def cargar_ranking(rutas):
    ranking = pd.read_csv(rutas["ranking"])

    if "Score_Final" not in ranking.columns:
        raise ValueError("No existe Score_Final en ranking.")

    return ranking


def normalizar_serie(s):
    s = pd.to_numeric(s, errors="coerce").fillna(0)

    minimo = s.min()
    maximo = s.max()

    if maximo - minimo == 0:
        return pd.Series(0.5, index=s.index)

    return (s - minimo) / (maximo - minimo)


def cargar_parametros_portfolio(nombre_mercado):
    cfg = get_json_config_value("portfolio", {})
    costs_global = cfg.get("costs", {}) if isinstance(cfg, dict) else {}

    market_cfg = {}
    if isinstance(cfg, dict):
        market_cfg = cfg.get("markets", {}).get(nombre_mercado, {})

    costs_market = market_cfg.get("costs", {}) if isinstance(market_cfg, dict) else {}
    merged_costs = {**costs_global, **costs_market}

    positions_cfg = {}
    if isinstance(market_cfg, dict) and isinstance(market_cfg.get("current_positions"), dict):
        positions_cfg = market_cfg.get("current_positions", {})
    elif isinstance(cfg, dict):
        # Backward compatibility with previous schema: portfolio.current_positions.<market>
        positions_cfg = cfg.get("current_positions", {}).get(nombre_mercado, {})

    capital_inicial_market = None
    if isinstance(market_cfg, dict):
        capital_inicial_market = market_cfg.get("capital_inicial")

    if capital_inicial_market is None:
        # Backward compatibility with previous schema
        capital_inicial_market = cfg.get("capital_inicial") if isinstance(cfg, dict) else None

    if capital_inicial_market is None:
        capital_inicial_market = CAPITAL_INICIAL

    params = {
        "capital_inicial": float(capital_inicial_market),
        "neutral_score": float(cfg.get("neutral_score", 50.0)) if isinstance(cfg, dict) else 50.0,
        "alpha_per_score_point_pct": float(cfg.get("alpha_per_score_point_pct", 0.03)) if isinstance(cfg, dict) else 0.03,
        "min_edge_buffer_pct": float(cfg.get("min_edge_buffer_pct", 0.20)) if isinstance(cfg, dict) else 0.20,
        "min_trade_value": float(cfg.get("min_trade_value", 250.0)) if isinstance(cfg, dict) else 250.0,
        "stop_loss_pct": float(cfg.get("stop_loss_pct", 0.015)) if isinstance(cfg, dict) else 0.015,
        "costs": {
            "buy_fee_pct": float(merged_costs.get("buy_fee_pct", 0.10)),
            "sell_fee_pct": float(merged_costs.get("sell_fee_pct", 0.10)),
            "slippage_pct": float(merged_costs.get("slippage_pct", 0.05)),
            "tax_sell_pct": float(merged_costs.get("tax_sell_pct", 0.00)),
            "fixed_cost_per_trade": float(merged_costs.get("fixed_cost_per_trade", 0.0)),
        },
        "current_positions": positions_cfg if isinstance(positions_cfg, dict) else {},
    }

    return params


def obtener_probabilidad(df):
    if "Consensus_Probability" in df.columns:
        prob = pd.to_numeric(df["Consensus_Probability"], errors="coerce").fillna(0)

        if prob.max() > 1:
            prob = prob / 100

        return prob.clip(0, 1)

    if "Probabilidad_IA" in df.columns:
        prob = pd.to_numeric(df["Probabilidad_IA"], errors="coerce").fillna(0)

        if prob.max() > 1:
            prob = prob / 100

        return prob.clip(0, 1)

    return pd.Series(0.5, index=df.index)


def obtener_riesgo(df):
    posibles = [
        "Volatilidad",
        "Volatility",
        "Riesgo",
        "Risk",
        "Drawdown",
        "Max_Drawdown",
    ]

    for col in posibles:
        if col in df.columns:
            return normalizar_serie(df[col])

    return pd.Series(0.5, index=df.index)


def obtener_momentum(df):
    posibles = [
        "Retorno_20d",
        "Return_20d",
        "Momentum",
        "Score_Tecnico",
        "Consensus_Score",
    ]

    for col in posibles:
        if col in df.columns:
            return normalizar_serie(df[col])

    return normalizar_serie(df["Score_Final"])


def factor_senal(senal):
    mapa = {
        "COMPRA FUERTE": 1.25,
        "COMPRA": 1.10,
        "VIGILAR": 0.90,
        "NEUTRAL": 0.60,
        "VENTA": 0.25,
        "VENTA FUERTE": 0.10,
    }

    return mapa.get(str(senal).strip().upper(), 0.50)


def preparar_ranking_portfolio(ranking, filtrar_senales=True):
    ranking = ranking.copy()

    if filtrar_senales and "Señal" in ranking.columns:
        ranking = ranking[
            ranking["Señal"].isin(["COMPRA FUERTE", "COMPRA", "VIGILAR"])
        ].copy()

    if ranking.empty:
        return ranking

    ranking["score_norm"] = normalizar_serie(ranking["Score_Final"])
    ranking["prob_norm"] = obtener_probabilidad(ranking)
    ranking["riesgo_norm"] = obtener_riesgo(ranking)
    ranking["momentum_norm"] = obtener_momentum(ranking)

    if "Señal" in ranking.columns:
        ranking["factor_senal"] = ranking["Señal"].apply(factor_senal)
    else:
        ranking["factor_senal"] = 0.75

    ranking["score_ajustado"] = (
        0.45 * ranking["score_norm"]
        + 0.25 * ranking["prob_norm"]
        + 0.20 * ranking["momentum_norm"]
        + 0.10 * ranking["factor_senal"]
        - 0.20 * ranking["riesgo_norm"]
    )

    ranking["score_ajustado"] = ranking["score_ajustado"].clip(lower=0)

    candidatos = ranking.sort_values("score_ajustado", ascending=False).head(
        len(ranking)
    )

    return candidatos.copy()


def seleccionar_activos(decision, ranking):
    max_posiciones = int(decision.get("Max_Posiciones", 0))

    if max_posiciones <= 0:
        return pd.DataFrame()

    candidatos = preparar_ranking_portfolio(ranking, filtrar_senales=True)

    if candidatos.empty:
        return candidatos

    return candidatos.head(max_posiciones).copy()


def construir_estado_actual(nombre_mercado, ranking, params):
    positions_cfg = params.get("current_positions", {})
    positions = positions_cfg.get("positions", []) if isinstance(positions_cfg, dict) else []
    cash = float(positions_cfg.get("cash", params["capital_inicial"])) if isinstance(positions_cfg, dict) else params["capital_inicial"]

    price_map = {}
    score_map = {}
    signal_map = {}
    name_map = {}

    if not ranking.empty:
        ranking_map = ranking.copy()
        ranking_map["_TickerNorm"] = ranking_map["Ticker"].apply(
            lambda x: ticker_para_lookup(x, nombre_mercado)
        )
        ranking_map = ranking_map.drop_duplicates("_TickerNorm", keep="first")

        if "Close" in ranking_map.columns:
            price_map = ranking_map.set_index("_TickerNorm")["Close"].to_dict()
        if "Score_Final" in ranking_map.columns:
            score_map = ranking_map.set_index("_TickerNorm")["Score_Final"].to_dict()
        if "Señal" in ranking_map.columns:
            signal_map = ranking_map.set_index("_TickerNorm")["Señal"].to_dict()
        if "Nombre" in ranking_map.columns:
            name_map = ranking_map.set_index("_TickerNorm")["Nombre"].to_dict()

    live_prices_enabled = usar_precios_en_vivo()

    filas = []

    for pos in positions:
        if not isinstance(pos, dict):
            continue

        ticker_raw = str(pos.get("ticker", "")).strip()
        ticker_norm = ticker_para_lookup(ticker_raw, nombre_mercado)
        if not ticker_norm:
            continue

        qty = float(pos.get("quantity", 0.0) or 0.0)
        avg_price = float(pos.get("avg_price", 0.0) or 0.0)

        current_price = float(price_map.get(ticker_norm, 0.0) or 0.0)
        if live_prices_enabled:
            live_price = _precio_en_vivo_yfinance(ticker_norm)
            if live_price is not None and live_price > 0:
                current_price = float(live_price)

        if current_price <= 0:
            current_price = avg_price

        market_value = float(pos.get("market_value", 0.0) or 0.0)
        if market_value <= 0 and qty > 0 and current_price > 0:
            market_value = qty * current_price

        filas.append({
            "Ticker": ticker_para_output(ticker_norm, nombre_mercado),
            "_TickerNorm": ticker_norm,
            "Nombre": pos.get("name") or name_map.get(ticker_norm, ticker_norm),
            "Quantity": qty,
            "Avg_Price": avg_price,
            "Current_Price": current_price,
            "Capital_Actual": market_value,
            "Score_Actual": float(score_map.get(ticker_norm, params["neutral_score"])),
            "Señal_Actual": signal_map.get(ticker_norm, ""),
        })

    cartera_actual = pd.DataFrame(filas)

    valor_actual = float(cartera_actual["Capital_Actual"].sum()) if not cartera_actual.empty else 0.0
    capital_total = cash + valor_actual

    if capital_total <= 0:
        capital_total = params["capital_inicial"]
        cash = capital_total

    return cartera_actual, round(cash, 2), round(capital_total, 2)


def limitar_concentracion(pesos):
    pesos = pesos.copy()

    for _ in range(10):
        exceso = pesos[pesos > MAX_PESO_ACTIVO] - MAX_PESO_ACTIVO

        if exceso.sum() <= 0:
            break

        pesos[pesos > MAX_PESO_ACTIVO] = MAX_PESO_ACTIVO

        redistribuir = exceso.sum()
        libres = pesos[pesos < MAX_PESO_ACTIVO]

        if libres.empty:
            break

        pesos.loc[libres.index] += redistribuir * libres / libres.sum()

    pesos = pesos / pesos.sum() * 100

    return pesos


def calcular_pesos(cartera):
    if cartera.empty:
        return cartera

    cartera = cartera.copy()

    base = cartera["score_ajustado"].clip(lower=0)

    if base.sum() == 0:
        cartera["Porcentaje"] = 100 / len(cartera)
    else:
        cartera["Porcentaje"] = base / base.sum() * 100


    if len(cartera) > 2:
        cartera["Porcentaje"] = limitar_concentracion(cartera["Porcentaje"])

    if len(cartera) > 3:
        cartera.loc[cartera["Porcentaje"] < MIN_PESO_ACTIVO, "Porcentaje"] = MIN_PESO_ACTIVO
        cartera["Porcentaje"] = cartera["Porcentaje"] / cartera["Porcentaje"].sum() * 100

    return cartera


def construir_cartera_objetivo(decision, ranking, cartera_actual, params, nombre_mercado):
    max_posiciones = int(decision.get("Max_Posiciones", 0))
    ticket_compra = float(params["min_trade_value"])

    if max_posiciones <= 0:
        return pd.DataFrame()

    ranking_full = preparar_ranking_portfolio(ranking, filtrar_senales=False)
    ranking_nuevo = preparar_ranking_portfolio(ranking, filtrar_senales=True)

    if ranking_full.empty:
        return pd.DataFrame()

    ranking_full = ranking_full.copy()
    ranking_full["_TickerNorm"] = ranking_full["Ticker"].apply(
        lambda x: ticker_para_lookup(x, nombre_mercado)
    )
    ranking_full = ranking_full.drop_duplicates("_TickerNorm", keep="first")

    ranking_nuevo = ranking_nuevo.copy()
    ranking_nuevo["_TickerNorm"] = ranking_nuevo["Ticker"].apply(
        lambda x: ticker_para_lookup(x, nombre_mercado)
    )
    ranking_nuevo = ranking_nuevo.drop_duplicates("_TickerNorm", keep="first")

    actuales_map = {}
    actuales_norm = set()
    if not cartera_actual.empty:
        actuales_map = cartera_actual.set_index("_TickerNorm")["Capital_Actual"].to_dict()
        actuales_norm = set(cartera_actual["_TickerNorm"].dropna().tolist())

    actuales_df = ranking_full[ranking_full["_TickerNorm"].isin(actuales_norm)].copy()
    actuales_df["es_actual"] = True
    actuales_df["Capital"] = actuales_df["_TickerNorm"].map(actuales_map).fillna(0.0)

    propuestas_df = ranking_nuevo[~ranking_nuevo["_TickerNorm"].isin(actuales_norm)].copy()
    propuestas_df = propuestas_df.head(max(max_posiciones, TOP_CANDIDATOS_EXTRA)).copy()
    propuestas_df["es_actual"] = False
    propuestas_df["Capital"] = ticket_compra

    comparables = pd.concat([actuales_df, propuestas_df], ignore_index=True, sort=False)
    if comparables.empty:
        return pd.DataFrame()

    comparables = comparables.drop_duplicates("_TickerNorm", keep="first")
    comparables.sort_values("score_ajustado", ascending=False, inplace=True)
    seleccion = comparables.head(max_posiciones).copy()

    columnas = [
        "Ticker",
        "Nombre",
        "Close",
        "Score_Final",
        "Señal",
        "score_ajustado",
        "_TickerNorm",
        "Capital",
        "es_actual",
    ]
    columnas = [c for c in columnas if c in seleccion.columns]
    return seleccion[columnas].copy()


def calcular_costo_operacion(monto, accion, params):
    costs = params["costs"]
    fixed = costs["fixed_cost_per_trade"]

    if accion == "BUY":
        pct = costs["buy_fee_pct"] + costs["slippage_pct"]
    else:
        pct = costs["sell_fee_pct"] + costs["slippage_pct"] + costs["tax_sell_pct"]

    costo = monto * (pct / 100.0) + fixed
    return round(costo, 2), pct


def generar_plan_rebalanceo(cartera_objetivo, cartera_actual, ranking, params, capital_total, nombre_mercado, cash_actual):
    neutral_score = params["neutral_score"]
    alpha_factor = params["alpha_per_score_point_pct"]
    edge_buffer = params["min_edge_buffer_pct"]
    min_trade_value = params["min_trade_value"]
    stop_loss_pct = float(params.get("stop_loss_pct", 0.0) or 0.0)
    live_prices_enabled = usar_precios_en_vivo()

    score_map = {}
    signal_map = {}
    close_map = {}
    name_map = {}

    if not ranking.empty:
        ranking_map = ranking.copy()
        ranking_map["_TickerNorm"] = ranking_map["Ticker"].apply(
            lambda x: ticker_para_lookup(x, nombre_mercado)
        )
        ranking_map = ranking_map.drop_duplicates("_TickerNorm", keep="first")

        if "Score_Final" in ranking_map.columns:
            score_map = ranking_map.set_index("_TickerNorm")["Score_Final"].to_dict()
        if "Señal" in ranking_map.columns:
            signal_map = ranking_map.set_index("_TickerNorm")["Señal"].to_dict()
        if "Close" in ranking_map.columns:
            close_map = ranking_map.set_index("_TickerNorm")["Close"].to_dict()
        if "Nombre" in ranking_map.columns:
            name_map = ranking_map.set_index("_TickerNorm")["Nombre"].to_dict()

    target_map = {}
    if not cartera_objetivo.empty:
        for _, row in cartera_objetivo.iterrows():
            ticker_raw = row.get("Ticker")
            ticker_norm = ticker_para_lookup(ticker_raw, nombre_mercado)
            if not ticker_norm:
                continue
            target_map[ticker_norm] = {
                "Ticker_Out": ticker_para_output(ticker_norm, nombre_mercado),
                "Capital_Objetivo": float(row.get("Capital", 0.0) or 0.0),
                "Porcentaje_Objetivo": float(row.get("Porcentaje", 0.0) or 0.0),
                "Score_Objetivo": float(row.get("Score_Final", score_map.get(ticker_norm, neutral_score))),
                "Señal_Objetivo": row.get("Señal", signal_map.get(ticker_norm, "")),
                "Nombre": row.get("Nombre", name_map.get(ticker_norm, ticker_norm)),
                "Close": float(row.get("Close", close_map.get(ticker_norm, 0.0) or 0.0) or 0.0),
            }

    current_map = {}
    if not cartera_actual.empty:
        for _, row in cartera_actual.iterrows():
            ticker_raw = row.get("Ticker")
            ticker_norm = row.get("_TickerNorm") or ticker_para_lookup(ticker_raw, nombre_mercado)
            if not ticker_norm:
                continue
            current_map[ticker_norm] = {
                "Ticker_Out": ticker_para_output(ticker_norm, nombre_mercado),
                "Capital_Actual": float(row.get("Capital_Actual", 0.0) or 0.0),
                "Score_Actual": float(row.get("Score_Actual", neutral_score)),
                "Señal_Actual": row.get("Señal_Actual", ""),
                "Nombre": row.get("Nombre", name_map.get(ticker_norm, ticker_norm)),
                "Avg_Price": float(row.get("Avg_Price", 0.0) or 0.0),
                "Current_Price": float(row.get("Current_Price", close_map.get(ticker_norm, 0.0) or 0.0) or 0.0),
                "Close": float(close_map.get(ticker_norm, 0.0) or 0.0),
            }

    best_target_score = neutral_score
    if target_map:
        best_target_score = max(v["Score_Objetivo"] for v in target_map.values())

    buy_leg_pct = params["costs"]["buy_fee_pct"] + params["costs"]["slippage_pct"]
    hay_nuevos_objetivo = any(
        (t not in current_map) and (v.get("Capital_Objetivo", 0.0) > 0.0)
        for t, v in target_map.items()
    )

    hay_compras_planificadas = any(
        float(target_map.get(ticker, {}).get("Capital_Objetivo", 0.0) or 0.0) >
        float(current_map.get(ticker, {}).get("Capital_Actual", 0.0) or 0.0)
        for ticker in set(target_map.keys()) | set(current_map.keys())
    )

    tickers = sorted(set(target_map.keys()) | set(current_map.keys()))
    plan = []

    for ticker in tickers:
        actual = current_map.get(ticker, {})
        objetivo = target_map.get(ticker, {})

        capital_actual = float(actual.get("Capital_Actual", 0.0) or 0.0)
        capital_objetivo = float(objetivo.get("Capital_Objetivo", 0.0) or 0.0)
        delta = capital_objetivo - capital_actual

        score_actual = float(actual.get("Score_Actual", neutral_score))
        score_objetivo = float(objetivo.get("Score_Objetivo", score_map.get(ticker, neutral_score)))

        senal_actual = actual.get("Señal_Actual", signal_map.get(ticker, ""))
        senal_objetivo = objetivo.get("Señal_Objetivo", signal_map.get(ticker, ""))

        nombre = objetivo.get("Nombre") or actual.get("Nombre") or name_map.get(ticker, ticker)
        close = objetivo.get("Close", 0.0) or actual.get("Close", 0.0) or float(close_map.get(ticker, 0.0) or 0.0)
        ticker_out = objetivo.get("Ticker_Out") or actual.get("Ticker_Out") or ticker_para_output(ticker, nombre_mercado)
        avg_price = float(actual.get("Avg_Price", 0.0) or 0.0)
        current_price = float(actual.get("Current_Price", close) or close or 0.0)

        stop_price = avg_price * (1.0 - stop_loss_pct) if avg_price > 0 and stop_loss_pct > 0 else 0.0
        stop_loss_triggered = (
            capital_actual > 0
            and avg_price > 0
            and current_price > 0
            and stop_loss_pct > 0
            and current_price <= stop_price
        )

        if stop_loss_triggered:
            capital_objetivo = 0.0
            delta = -capital_actual

        accion_deseada = "HOLD"
        monto_trade = abs(delta)
        if delta > 0:
            accion_deseada = "BUY"
        elif delta < 0:
            accion_deseada = "SELL"

        costo_estimado = 0.0
        costo_pct = 0.0
        required_pct = 0.0
        edge_estimado_pct = 0.0
        ventaja_neta_pct = 0.0
        ejecutar = False
        accion_final = "HOLD"
        motivo = "Sin cambios relevantes."

        if stop_loss_triggered and accion_deseada == "SELL":
            costo_estimado, costo_pct = calcular_costo_operacion(monto_trade, accion_deseada, params)
            ejecutar = True
            accion_final = "STOP_LOSS_EXIT"
            motivo = (
                f"Salida por stop-loss: precio actual {current_price:.8f} <= "
                f"límite {stop_price:.8f} ({stop_loss_pct * 100:.2f}% debajo del promedio)."
            )
        elif accion_deseada in ("BUY", "SELL"):
            costo_estimado, costo_pct = calcular_costo_operacion(monto_trade, accion_deseada, params)

            if accion_deseada == "BUY":
                score_gap = max(0.0, score_objetivo - score_actual)
                edge_estimado_pct = score_gap * alpha_factor

                if live_prices_enabled and current_price > 0 and close > 0:
                    live_upside_pct = max(0.0, ((current_price - close) / close) * 100.0)
                    if live_upside_pct < edge_estimado_pct:
                        edge_estimado_pct = live_upside_pct

                required_pct = costo_pct + edge_buffer
                motivo = "Compra sugerida si la mejora esperada supera costos de entrada."
            else:
                replacement_score = best_target_score
                score_gap = max(0.0, replacement_score - score_actual)
                edge_estimado_pct = score_gap * alpha_factor
                required_pct = costo_pct + edge_buffer

                # Si para esta rotación hay activos nuevos objetivo, exige costo de salida + entrada.
                if hay_nuevos_objetivo:
                    required_pct = costo_pct + buy_leg_pct + edge_buffer

                motivo = "Venta sugerida si migrar capital a alternativas mejora el neto de costos."

            ventaja_neta_pct = edge_estimado_pct - required_pct
            ejecutar = ventaja_neta_pct > 0

            if ejecutar:
                if accion_deseada == "BUY" and capital_actual > 0:
                    accion_final = "INCREASE"
                elif accion_deseada == "SELL" and capital_objetivo > 0:
                    accion_final = "REDUCE"
                elif accion_deseada == "SELL" and capital_objetivo == 0:
                    accion_final = "SWITCH_OUT"
                elif accion_deseada == "BUY" and capital_actual == 0:
                    accion_final = "SWITCH_IN"
                else:
                    accion_final = accion_deseada
            else:
                accion_final = "HOLD"
                motivo = (
                    f"Mantener: mejora esperada {edge_estimado_pct:.2f}% <= "
                    f"costo+buffer {required_pct:.2f}%."
                )

        plan.append({
            "Ticker": ticker_out,
            "Nombre": nombre,
            "Close": close,
            "Current_Price": round(current_price, 8),
            "Capital_Actual": round(capital_actual, 2),
            "Capital_Objetivo": round(capital_objetivo, 2),
            "Delta_Capital": round(delta, 2),
            "Accion_Deseada": accion_deseada,
            "Accion_Sugerida": accion_final,
            "Ejecutar": bool(ejecutar),
            "Costo_Estimado": round(costo_estimado, 2),
            "Costo_pct": round(costo_pct, 4),
            "Required_pct": round(required_pct, 4),
            "Edge_Estimado_pct": round(edge_estimado_pct, 4),
            "Ventaja_Neta_pct": round(ventaja_neta_pct, 4),
            "Score_Actual": round(score_actual, 2),
            "Score_Objetivo": round(score_objetivo, 2),
            "Señal_Actual": senal_actual,
            "Señal_Objetivo": senal_objetivo,
            "Switch_Target": "",
            "Switch_Target_Score": 0.0,
            "Stop_Loss_Triggered": bool(stop_loss_triggered),
            "Motivo": motivo,
        })

    plan_df = pd.DataFrame(plan)
    if not plan_df.empty:
        capital_total_compras_plan = float(plan_df.loc[plan_df["Accion_Deseada"] == "BUY", "Delta_Capital"].sum())
        capital_total_ventas_ejecutables = float(
            (-plan_df.loc[
                (plan_df["Accion_Deseada"] == "SELL") & (plan_df["Ejecutar"] == True),  # noqa: E712
                "Delta_Capital",
            ]).sum()
        )
        capital_disponible_para_compras = float(cash_actual) + capital_total_ventas_ejecutables

        buys_ejecutables = plan_df[
            (plan_df["Accion_Deseada"] == "BUY") & (plan_df["Ejecutar"] == True)  # noqa: E712
        ].copy()
        if not buys_ejecutables.empty:
            buys_ejecutables.sort_values(["Ventaja_Neta_pct", "Score_Objetivo"], ascending=[False, False], inplace=True)
            acumulado = 0.0
            permitidos = []

            for idx, row in buys_ejecutables.iterrows():
                delta = float(row.get("Delta_Capital", 0.0) or 0.0)
                if acumulado + delta <= capital_disponible_para_compras + 1e-9:
                    permitidos.append(idx)
                    acumulado += delta

            bloqueados = [idx for idx in buys_ejecutables.index.tolist() if idx not in permitidos]
            if bloqueados:
                plan_df.loc[bloqueados, "Ejecutar"] = False
                plan_df.loc[bloqueados, "Accion_Sugerida"] = "HOLD"
                plan_df.loc[bloqueados, "Motivo"] = (
                    f"Compra no fondeada: cash+ventas ejecutables ${capital_disponible_para_compras:,.2f} "
                    f"no alcanza para todas las propuestas de ${capital_total_compras_plan:,.2f}."
                )

        buys_financiadas = plan_df[
            (plan_df["Accion_Deseada"] == "BUY") & (plan_df["Ejecutar"] == True)  # noqa: E712
        ].copy()
        sells_candidatas = plan_df[
            (plan_df["Accion_Deseada"] == "SELL") & (~plan_df["Stop_Loss_Triggered"].fillna(False))
        ].copy()

        if not buys_financiadas.empty and not sells_candidatas.empty:
            buys_financiadas.sort_values(["Score_Objetivo", "Ventaja_Neta_pct"], ascending=[False, False], inplace=True)
            sells_candidatas.sort_values(["Score_Actual", "Capital_Actual"], ascending=[True, False], inplace=True)

            pares = list(zip(sells_candidatas.index.tolist(), buys_financiadas.index.tolist()))
            for sell_idx, buy_idx in pares:
                buy_ticker = str(plan_df.at[buy_idx, "Ticker"])
                buy_score = float(plan_df.at[buy_idx, "Score_Objetivo"])
                sell_score = float(plan_df.at[sell_idx, "Score_Actual"])
                sell_costo_pct = float(plan_df.at[sell_idx, "Costo_pct"])
                required_pct = sell_costo_pct + buy_leg_pct + edge_buffer
                edge_estimado_pct = max(0.0, buy_score - sell_score) * alpha_factor
                ventaja_neta_pct = edge_estimado_pct - required_pct
                ejecutar_sell = ventaja_neta_pct > 0

                plan_df.at[sell_idx, "Switch_Target"] = buy_ticker
                plan_df.at[sell_idx, "Switch_Target_Score"] = round(buy_score, 2)
                plan_df.at[sell_idx, "Required_pct"] = round(required_pct, 4)
                plan_df.at[sell_idx, "Edge_Estimado_pct"] = round(edge_estimado_pct, 4)
                plan_df.at[sell_idx, "Ventaja_Neta_pct"] = round(ventaja_neta_pct, 4)
                plan_df.at[sell_idx, "Ejecutar"] = bool(ejecutar_sell)

                if ejecutar_sell:
                    plan_df.at[sell_idx, "Accion_Sugerida"] = "SWITCH_OUT"
                    plan_df.at[sell_idx, "Motivo"] = (
                        f"Venta sugerida para migrar desde {plan_df.at[sell_idx, 'Ticker']} hacia {buy_ticker} "
                        f"si la mejora neta {ventaja_neta_pct:.2f}% supera costos totales."
                    )
                else:
                    plan_df.at[sell_idx, "Accion_Sugerida"] = "HOLD"
                    plan_df.at[sell_idx, "Motivo"] = (
                        f"Mantener: {buy_ticker} mejora {edge_estimado_pct:.2f}% <= costo total {required_pct:.2f}% "
                        f"para salir y entrar."
                    )

            paired_sells = {sell_idx for sell_idx, _ in pares}
            sells_sin_par = [idx for idx in sells_candidatas.index.tolist() if idx not in paired_sells]
            if sells_sin_par:
                plan_df.loc[sells_sin_par, "Ejecutar"] = False
                plan_df.loc[sells_sin_par, "Accion_Sugerida"] = "HOLD"
                plan_df.loc[sells_sin_par, "Motivo"] = "Mantener: no hay una compra financiada asociada para justificar la salida."

        if not (plan_df["Accion_Deseada"] == "BUY").any() or not (plan_df["Ejecutar"] & (plan_df["Accion_Deseada"] == "BUY")).any():
            mask_sell = (plan_df["Accion_Deseada"] == "SELL") & (~plan_df["Stop_Loss_Triggered"].fillna(False))
            plan_df.loc[mask_sell, "Ejecutar"] = False
            plan_df.loc[mask_sell, "Accion_Sugerida"] = "HOLD"
            plan_df.loc[mask_sell, "Motivo"] = plan_df.loc[mask_sell, "Motivo"].where(
                plan_df.loc[mask_sell, "Ventaja_Neta_pct"] > 0,
                plan_df.loc[mask_sell, "Motivo"],
            )

        plan_df["Trade_Abs"] = plan_df["Delta_Capital"].abs()
        plan_df.sort_values(["Ejecutar", "Trade_Abs"], ascending=[False, False], inplace=True)
        plan_df.drop(columns=["Trade_Abs"], inplace=True)

    return plan_df


def construir_portafolio(nombre_mercado="merval"):
    inicio = datetime.now()

    rutas = rutas_portfolio(nombre_mercado)
    market = rutas["market"]

    print("=" * 80)
    print(f"PORTFOLIO OPTIMIZER - {market.display_name}")
    print("=" * 80)

    verificar_archivos(rutas)

    decision = cargar_decision(rutas)
    ranking = cargar_ranking(rutas)
    params = cargar_parametros_portfolio(market.market)

    cartera_actual, cash_actual, capital_total = construir_estado_actual(
        market.market,
        ranking,
        params,
    )

    cartera = construir_cartera_objetivo(
        decision=decision,
        ranking=ranking,
        cartera_actual=cartera_actual,
        params=params,
        nombre_mercado=market.market,
    )
    invertido_objetivo = float(cartera["Capital"].sum()) if not cartera.empty and "Capital" in cartera.columns else 0.0
    cash_objetivo = max(0.0, capital_total - invertido_objetivo)

    plan = generar_plan_rebalanceo(
        cartera_objetivo=cartera,
        cartera_actual=cartera_actual,
        ranking=ranking,
        params=params,
        capital_total=capital_total,
        nombre_mercado=market.market,
        cash_actual=cash_actual,
    )
    capital_total_compras = float(plan.loc[plan["Delta_Capital"] > 0, "Delta_Capital"].sum()) if not plan.empty else 0.0

    exportar = plan.copy()

    exportar.to_csv(rutas["csv_out"], index=False, encoding="utf-8-sig")
    exportar.to_csv(rutas["csv_ts_out"], index=False, encoding="utf-8-sig")
    excel_resumen = "OMITIDO (usar --excel-all para habilitar)"
    if exportar_excel_portfolio_habilitado():
        exportar.to_excel(rutas["xlsx_out"], index=False)
        excel_resumen = str(rutas["xlsx_out"])

    sugeridas = plan[(plan["Ejecutar"] == True) & (plan["Accion_Sugerida"] != "HOLD")].copy()  # noqa: E712
    costo_total_sugerido = float(sugeridas["Costo_Estimado"].sum()) if not sugeridas.empty else 0.0
    stay = sugeridas.empty

    with open(rutas["txt_out"], "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"MARKET AI PORTFOLIO OPTIMIZER - {market.display_name}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Fecha                 : {decision.get('Fecha', '')}\n")
        f.write(f"Mercado               : {market.display_name}\n")
        f.write(f"Régimen               : {decision.get('Regimen', '')}\n")
        f.write(f"Decisión              : {decision.get('Decision', '')}\n")
        f.write(f"Capital total actual  : ${capital_total:,.2f}\n")
        f.write(f"Cash actual           : ${cash_actual:,.2f}\n")
        f.write(f"Cash objetivo         : ${cash_objetivo:,.2f}\n")
        f.write(f"Capital activos obj.  : ${invertido_objetivo:,.2f}\n")
        f.write(f"Posiciones actuales   : {len(cartera_actual)}\n")
        f.write(f"Posiciones objetivo   : {len(cartera)}\n\n")

        f.write("-" * 80 + "\n")
        f.write("COSTOS OPERATIVOS CONFIGURADOS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Buy fee (%)           : {params['costs']['buy_fee_pct']:.4f}\n")
        f.write(f"Sell fee (%)          : {params['costs']['sell_fee_pct']:.4f}\n")
        f.write(f"Slippage (%)          : {params['costs']['slippage_pct']:.4f}\n")
        f.write(f"Tax sell (%)          : {params['costs']['tax_sell_pct']:.4f}\n")
        f.write(f"Costo fijo por trade  : ${params['costs']['fixed_cost_per_trade']:,.2f}\n")
        f.write(f"Buffer edge mínimo(%) : {params['min_edge_buffer_pct']:.4f}\n")
        f.write(f"Stop-loss (%)         : {params['stop_loss_pct'] * 100:.4f}\n")
        f.write(f"Ticket por compra ($) : ${params['min_trade_value']:,.2f}\n")
        f.write(f"Compras objetivo($)   : ${capital_total_compras:,.2f}\n\n")

        f.write("-" * 80 + "\n")
        f.write("ACCIONES SUGERIDAS (COSTO-AWARE)\n")
        f.write("-" * 80 + "\n")

        if stay:
            f.write("Recomendación: MANTENER POSICIONES ACTUALES.\n")
            f.write("Ningún cambio supera el umbral de mejora neta después de costos.\n")
        else:
            for _, row in sugeridas.iterrows():
                f.write(
                    f"{row.get('Ticker', ''):12} | "
                    f"{str(row.get('Nombre', ''))[:30]:30} | "
                    f"{row.get('Accion_Sugerida', ''):10} | "
                    f"vs {str(row.get('Switch_Target', '') or '-'):12} | "
                    f"Delta ${row.get('Delta_Capital', 0):11,.2f} | "
                    f"Costo ${row.get('Costo_Estimado', 0):9,.2f} | "
                    f"EdgeNet {row.get('Ventaja_Neta_pct', 0):7.3f}%\n"
                )

        f.write("\n")
        f.write(f"Costo total sugerido  : ${costo_total_sugerido:,.2f}\n")

    print()
    print("=" * 80)
    print("PORTAFOLIO / REBALANCEO SUGERIDO")
    print("=" * 80)

    print(f"Mercado         : {market.display_name}")
    print(f"CSV timestamped : {rutas['csv_ts_out'].name}")
    print(f"Capital total   : ${capital_total:,.2f}")
    print(f"Cash actual     : ${cash_actual:,.2f}")
    print(f"Cash objetivo   : ${cash_objetivo:,.2f}")
    print(f"Invertido obj.  : ${invertido_objetivo:,.2f}")
    print(f"Posiciones act. : {len(cartera_actual)}")
    print(f"Posiciones obj. : {len(cartera)}")
    print()

    if stay:
        print("Recomendación: MANTENER posiciones actuales.")
        print("No hay trades con ventaja neta positiva luego de costos.")
    else:
        mostrar = [
            "Ticker",
            "Nombre",
            "Accion_Sugerida",
            "Switch_Target",
            "Capital_Actual",
            "Capital_Objetivo",
            "Delta_Capital",
            "Costo_Estimado",
            "Edge_Estimado_pct",
            "Ventaja_Neta_pct",
            "Motivo",
        ]

        mostrar = [c for c in mostrar if c in sugeridas.columns]
        print(sugeridas[mostrar].to_string(index=False))

    print()
    print(f"Costo total trades sugeridos: ${costo_total_sugerido:,.2f}")

    print()
    print("=" * 80)
    print("ARCHIVOS GENERADOS")
    print("=" * 80)
    print(rutas["csv_out"])
    print(excel_resumen)
    print(rutas["txt_out"])

    print()
    print("=" * 80)
    print("PORTFOLIO OPTIMIZER FINALIZADO")
    print("=" * 80)
    print(f"Tiempo total : {datetime.now() - inicio}")

    return exportar


if __name__ == "__main__":
    construir_portafolio("crypto")