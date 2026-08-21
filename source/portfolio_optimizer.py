# portfolio_optimizer.py

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

from source.app_logger import log_print as print

from config import MarketConfig, get_json_config_value


CAPITAL_INICIAL = 100000.0

MAX_PESO_ACTIVO = 45.0
MIN_PESO_ACTIVO = 8.0
TOP_CANDIDATOS_EXTRA = 8


def rutas_portfolio(nombre_mercado):
    market = MarketConfig(nombre_mercado)
    reportes_dir = ROOT / "reportes" / market.market

    return {
        "market": market,
        "ranking": reportes_dir / "ranking_diario.csv",
        "decision": reportes_dir / f"decision_engine_{market.market}.csv",
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


def seleccionar_activos(decision, ranking):
    max_posiciones = int(decision.get("Max_Posiciones", 0))

    if max_posiciones <= 0:
        return pd.DataFrame()

    ranking = ranking.copy()

    if "Señal" in ranking.columns:
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
        max(max_posiciones, TOP_CANDIDATOS_EXTRA)
    )

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
        if "Close" in ranking.columns:
            price_map = ranking.set_index("Ticker")["Close"].to_dict()
        if "Score_Final" in ranking.columns:
            score_map = ranking.set_index("Ticker")["Score_Final"].to_dict()
        if "Señal" in ranking.columns:
            signal_map = ranking.set_index("Ticker")["Señal"].to_dict()
        if "Nombre" in ranking.columns:
            name_map = ranking.set_index("Ticker")["Nombre"].to_dict()

    filas = []

    for pos in positions:
        if not isinstance(pos, dict):
            continue

        ticker = str(pos.get("ticker", "")).strip()
        if not ticker:
            continue

        qty = float(pos.get("quantity", 0.0) or 0.0)
        avg_price = float(pos.get("avg_price", 0.0) or 0.0)

        current_price = float(pos.get("current_price", 0.0) or 0.0)
        if current_price <= 0:
            current_price = float(price_map.get(ticker, 0.0) or 0.0)
        if current_price <= 0:
            current_price = avg_price

        market_value = float(pos.get("market_value", 0.0) or 0.0)
        if market_value <= 0 and qty > 0 and current_price > 0:
            market_value = qty * current_price

        filas.append({
            "Ticker": ticker,
            "Nombre": pos.get("name") or name_map.get(ticker, ticker),
            "Quantity": qty,
            "Avg_Price": avg_price,
            "Current_Price": current_price,
            "Capital_Actual": market_value,
            "Score_Actual": float(score_map.get(ticker, params["neutral_score"])),
            "Señal_Actual": signal_map.get(ticker, ""),
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


def distribuir_capital(decision, cartera, capital_total):
    cash_pct = float(decision.get("Cash_pct", 100))
    activos_pct = float(decision.get("Acciones_pct", 0))

    capital_cash = capital_total * cash_pct / 100
    capital_activos = capital_total * activos_pct / 100

    if cartera.empty:
        return cartera, round(capital_cash, 2), round(capital_activos, 2)

    cartera = cartera.copy()
    cartera["Capital"] = capital_activos * cartera["Porcentaje"] / 100

    return cartera, round(capital_cash, 2), round(capital_activos, 2)


def calcular_costo_operacion(monto, accion, params):
    costs = params["costs"]
    fixed = costs["fixed_cost_per_trade"]

    if accion == "BUY":
        pct = costs["buy_fee_pct"] + costs["slippage_pct"]
    else:
        pct = costs["sell_fee_pct"] + costs["slippage_pct"] + costs["tax_sell_pct"]

    costo = monto * (pct / 100.0) + fixed
    return round(costo, 2), pct


def generar_plan_rebalanceo(cartera_objetivo, cartera_actual, ranking, params, capital_total):
    neutral_score = params["neutral_score"]
    alpha_factor = params["alpha_per_score_point_pct"]
    edge_buffer = params["min_edge_buffer_pct"]
    min_trade_value = params["min_trade_value"]

    score_map = {}
    signal_map = {}
    close_map = {}
    name_map = {}

    if not ranking.empty:
        if "Score_Final" in ranking.columns:
            score_map = ranking.set_index("Ticker")["Score_Final"].to_dict()
        if "Señal" in ranking.columns:
            signal_map = ranking.set_index("Ticker")["Señal"].to_dict()
        if "Close" in ranking.columns:
            close_map = ranking.set_index("Ticker")["Close"].to_dict()
        if "Nombre" in ranking.columns:
            name_map = ranking.set_index("Ticker")["Nombre"].to_dict()

    target_map = {}
    if not cartera_objetivo.empty:
        for _, row in cartera_objetivo.iterrows():
            ticker = row.get("Ticker")
            if not ticker:
                continue
            target_map[ticker] = {
                "Capital_Objetivo": float(row.get("Capital", 0.0) or 0.0),
                "Porcentaje_Objetivo": float(row.get("Porcentaje", 0.0) or 0.0),
                "Score_Objetivo": float(row.get("Score_Final", score_map.get(ticker, neutral_score))),
                "Señal_Objetivo": row.get("Señal", signal_map.get(ticker, "")),
                "Nombre": row.get("Nombre", name_map.get(ticker, ticker)),
                "Close": float(row.get("Close", close_map.get(ticker, 0.0) or 0.0) or 0.0),
            }

    current_map = {}
    if not cartera_actual.empty:
        for _, row in cartera_actual.iterrows():
            ticker = row.get("Ticker")
            if not ticker:
                continue
            current_map[ticker] = {
                "Capital_Actual": float(row.get("Capital_Actual", 0.0) or 0.0),
                "Score_Actual": float(row.get("Score_Actual", neutral_score)),
                "Señal_Actual": row.get("Señal_Actual", ""),
                "Nombre": row.get("Nombre", name_map.get(ticker, ticker)),
                "Close": float(close_map.get(ticker, 0.0) or 0.0),
            }

    best_target_score = neutral_score
    if target_map:
        best_target_score = max(v["Score_Objetivo"] for v in target_map.values())

    tickers = sorted(set(target_map.keys()) | set(current_map.keys()))
    plan = []

    for ticker in tickers:
        actual = current_map.get(ticker, {})
        objetivo = target_map.get(ticker, {})

        capital_actual = float(actual.get("Capital_Actual", 0.0) or 0.0)
        capital_objetivo = float(objetivo.get("Capital_Objetivo", 0.0) or 0.0)
        delta = capital_objetivo - capital_actual

        score_actual = float(actual.get("Score_Actual", score_map.get(ticker, neutral_score)))
        score_objetivo = float(objetivo.get("Score_Objetivo", score_map.get(ticker, neutral_score)))

        senal_actual = actual.get("Señal_Actual", signal_map.get(ticker, ""))
        senal_objetivo = objetivo.get("Señal_Objetivo", signal_map.get(ticker, ""))

        nombre = objetivo.get("Nombre") or actual.get("Nombre") or name_map.get(ticker, ticker)
        close = objetivo.get("Close", 0.0) or actual.get("Close", 0.0) or float(close_map.get(ticker, 0.0) or 0.0)

        accion_deseada = "HOLD"
        monto_trade = abs(delta)
        if delta > min_trade_value:
            accion_deseada = "BUY"
        elif delta < -min_trade_value:
            accion_deseada = "SELL"

        costo_estimado = 0.0
        costo_pct = 0.0
        edge_estimado_pct = 0.0
        ventaja_neta_pct = 0.0
        ejecutar = False
        accion_final = "HOLD"
        motivo = "Sin cambios relevantes."

        if accion_deseada in ("BUY", "SELL"):
            costo_estimado, costo_pct = calcular_costo_operacion(monto_trade, accion_deseada, params)

            if accion_deseada == "BUY":
                score_gap = max(0.0, score_objetivo - score_actual)
                edge_estimado_pct = score_gap * alpha_factor
                motivo = "Compra sugerida si la mejora esperada supera costos de entrada."
            else:
                replacement_score = best_target_score
                score_gap = max(0.0, replacement_score - score_actual)
                edge_estimado_pct = score_gap * alpha_factor
                motivo = "Venta sugerida si migrar capital a alternativas mejora el neto de costos."

            required_pct = costo_pct + edge_buffer
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
            "Ticker": ticker,
            "Nombre": nombre,
            "Close": close,
            "Capital_Actual": round(capital_actual, 2),
            "Capital_Objetivo": round(capital_objetivo, 2),
            "Delta_Capital": round(delta, 2),
            "Accion_Deseada": accion_deseada,
            "Accion_Sugerida": accion_final,
            "Ejecutar": bool(ejecutar),
            "Costo_Estimado": round(costo_estimado, 2),
            "Costo_pct": round(costo_pct, 4),
            "Edge_Estimado_pct": round(edge_estimado_pct, 4),
            "Ventaja_Neta_pct": round(ventaja_neta_pct, 4),
            "Score_Actual": round(score_actual, 2),
            "Score_Objetivo": round(score_objetivo, 2),
            "Señal_Actual": senal_actual,
            "Señal_Objetivo": senal_objetivo,
            "Motivo": motivo,
        })

    plan_df = pd.DataFrame(plan)
    if not plan_df.empty:
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

    cartera = seleccionar_activos(decision, ranking)
    cartera = calcular_pesos(cartera)
    cartera, cash_objetivo, invertido_objetivo = distribuir_capital(decision, cartera, capital_total)

    plan = generar_plan_rebalanceo(
        cartera_objetivo=cartera,
        cartera_actual=cartera_actual,
        ranking=ranking,
        params=params,
        capital_total=capital_total,
    )

    exportar = plan.copy()

    exportar.to_csv(rutas["csv_out"], index=False, encoding="utf-8-sig")
    exportar.to_excel(rutas["xlsx_out"], index=False)

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
        f.write(f"Trade mínimo ($)      : ${params['min_trade_value']:,.2f}\n\n")

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
    print(rutas["xlsx_out"])
    print(rutas["txt_out"])

    print()
    print("=" * 80)
    print("PORTFOLIO OPTIMIZER FINALIZADO")
    print("=" * 80)
    print(f"Tiempo total : {datetime.now() - inicio}")

    return exportar


if __name__ == "__main__":
    construir_portafolio("crypto")