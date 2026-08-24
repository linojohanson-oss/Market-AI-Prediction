from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

from source.app_logger import log_print as print
from source.market_downloader import descargar_mercado
from source.market_indicators import generar_indicadores
from source.market_dataset import generar_dataset_ia
from source.market_training import entrenar_mercado
from source.market_model_manager import MarketModelManager
from source import market_scoring as scoring
from source import portfolio_optimizer as optimizer
from config import get_json_config_value


def _load_numeric_list_config(path: str, fallback: List[float], cast_type):
    values = get_json_config_value(path, None)
    if not isinstance(values, list) or not values:
        return list(fallback)

    normalized = []
    for value in values:
        try:
            normalized.append(cast_type(value))
        except (TypeError, ValueError):
            continue

    return normalized or list(fallback)


@dataclass
class SimulationConfig:
    markets: List[str] = None
    target_days_list: List[int] = None
    target_min_returns: List[float] = None
    stop_loss_pct_list: List[float] = None
    start_date: str = "2026-01-01"
    end_date: str = "2026-08-15"
    liquidation_date: str = "2026-08-15"
    initial_capital: float = 100000.0
    max_positions: int = 5
    run_pipeline_per_scenario: bool = True
    top_models_for_scoring: int = 3
    output_dir: Path = ROOT / "reportes" / "simulaciones_optimizer"

    def __post_init__(self):
        default_target_days = _load_numeric_list_config(
            "optimizer_simulator.target_days_list",
            [5, 10, 15],
            int,
        )
        default_target_returns = _load_numeric_list_config(
            "optimizer_simulator.target_min_returns",
            [0.01, 0.015, 0.03, 0.05],
            float,
        )
        default_stop_losses = _load_numeric_list_config(
            "optimizer_simulator.stop_loss_pct_list",
            [0.015, 0.02, 0.03],
            float,
        )

        if self.markets is None:
            self.markets = ["crypto"]
        if self.target_days_list is None:
            self.target_days_list = sorted(set(default_target_days))
        if self.target_min_returns is None:
            self.target_min_returns = sorted(set(default_target_returns))
        if self.stop_loss_pct_list is None:
            self.stop_loss_pct_list = sorted(set(default_stop_losses))


def _scenario_id(market: str, target_days: int, min_return: float, stop_loss_pct: float) -> str:
    pct = f"{min_return * 100:.2f}".replace(".", "p")
    stop = f"{stop_loss_pct * 100:.2f}".replace(".", "p")
    return f"{market}_t{target_days}d_r{pct}_sl{stop}"


def _ret_label(min_return: float) -> str:
    return f"{min_return * 100:.2f}".replace(".", "p")


def _stop_label(stop_loss_pct: float) -> str:
    return f"{stop_loss_pct * 100:.2f}".replace(".", "p")


def _prepare_scenario_artifacts(market: str, target_days: int, target_min_return: float) -> None:
    import source.market_indicators as indicators_module

    target_set = sorted(set(list(indicators_module.TARGETS) + [int(target_days)]))
    indicators_module.TARGETS = target_set
    indicators_module.TARGET_THRESHOLD_DAYS = int(target_days)
    indicators_module.TARGET_MIN_RETURN = float(target_min_return)

    target_principal = f"Target_{int(target_days)}d"

    descargar_mercado(market)
    generar_indicadores(market)
    generar_dataset_ia(
        market,
        target_principal=target_principal,
        target_horizons=target_set,
    )
    entrenar_mercado(market)


def _calcular_consenso_diario(
    day_df: pd.DataFrame,
    modelos: List[Dict],
    features: List[str],
    top_models: int,
) -> pd.DataFrame:
    modelos = sorted(modelos, key=lambda x: x.get("peso", 0), reverse=True)[:top_models]
    base = day_df.copy()

    X = base[features].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    matriz_probs = []
    pesos = []

    for m in modelos:
        prob = scoring.probabilidad_modelo(m["objeto"], X)
        matriz_probs.append(prob)
        pesos.append(float(m.get("peso", 0.0)))

    if not matriz_probs:
        raise RuntimeError("No se pudieron calcular probabilidades para el escenario.")

    matriz_probs = np.vstack(matriz_probs).T
    pesos = np.array(pesos, dtype=float)

    if pesos.sum() <= 0:
        pesos = np.ones_like(pesos) / len(pesos)
    else:
        pesos = pesos / pesos.sum()

    consensus_probability = np.average(matriz_probs, axis=1, weights=pesos)

    rel_scores = []
    for i in range(matriz_probs.shape[1]):
        serie = pd.Series(matriz_probs[:, i])
        rel_scores.append(serie.rank(pct=True).values * 100)
    rel_scores = np.vstack(rel_scores).T

    score_relativo = np.average(rel_scores, axis=1, weights=pesos)
    score_absoluto = consensus_probability * 100
    consensus_score = score_relativo * 0.70 + score_absoluto * 0.30

    dispersion = np.std(rel_scores, axis=1)
    agreement_score = 100 - dispersion

    percentiles = np.percentile(rel_scores, [33, 66], axis=0)
    umbral_venta = percentiles[0]
    umbral_compra = percentiles[1]

    votos_compra = (rel_scores >= umbral_compra).sum(axis=1)
    votos_venta = (rel_scores <= umbral_venta).sum(axis=1)
    votos_neutral = rel_scores.shape[1] - votos_compra - votos_venta

    base["Consensus_Probability"] = consensus_probability
    base["Score_Relativo"] = score_relativo
    base["Score_Absoluto"] = score_absoluto
    base["Consensus_Score"] = consensus_score
    base["Model_Dispersion"] = dispersion
    base["Agreement_Score"] = agreement_score
    base["Uncertainty_Score"] = dispersion
    base["Buy_Votes"] = votos_compra
    base["Neutral_Votes"] = votos_neutral
    base["Sell_Votes"] = votos_venta
    base["Models_Used"] = rel_scores.shape[1]

    base = scoring.calcular_score_tecnico(base)
    base = scoring.clasificar_senal(base)
    base.sort_values("Score_Final", ascending=False, inplace=True)

    return base


def _build_daily_ranking_frame(
    market: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    top_models: int,
) -> pd.DataFrame:
    manager = MarketModelManager(market)
    manager.guardar_modelo_activo()
    manager.guardar_ensemble()

    dataset = manager.cargar_dataset()
    dataset["Date"] = pd.to_datetime(dataset["Date"], errors="coerce")
    dataset = dataset.dropna(subset=["Date"]).copy()

    mask = (dataset["Date"] >= start_date) & (dataset["Date"] <= end_date)
    dataset = dataset.loc[mask].copy()

    if dataset.empty:
        raise RuntimeError("No hay datos en el rango solicitado para simular.")

    features = manager.cargar_features()
    modelos = manager.cargar_modelos_ensemble()

    rankings = []
    for fecha, day_df in dataset.groupby("Date", sort=True):
        day_rank = _calcular_consenso_diario(day_df.copy(), modelos, features, top_models)
        day_rank["Date"] = fecha
        rankings.append(day_rank)

    all_days = pd.concat(rankings, ignore_index=True)
    all_days.sort_values(["Date", "Score_Final"], ascending=[True, False], inplace=True)

    return all_days


def _build_cartera_actual(
    positions: Dict[str, Dict[str, float]],
    ranking_day: pd.DataFrame,
    market: str,
    neutral_score: float,
) -> pd.DataFrame:
    if not positions:
        return pd.DataFrame(columns=[
            "Ticker",
            "_TickerNorm",
            "Nombre",
            "Quantity",
            "Avg_Price",
            "Current_Price",
            "Capital_Actual",
            "Score_Actual",
            "Señal_Actual",
        ])

    rank = ranking_day.copy()
    rank["_TickerNorm"] = rank["Ticker"].apply(lambda x: optimizer.ticker_para_lookup(x, market))
    rank = rank.drop_duplicates("_TickerNorm", keep="first")

    close_map = rank.set_index("_TickerNorm")["Close"].to_dict() if "Close" in rank.columns else {}
    score_map = rank.set_index("_TickerNorm")["Score_Final"].to_dict() if "Score_Final" in rank.columns else {}
    signal_map = rank.set_index("_TickerNorm")["Señal"].to_dict() if "Señal" in rank.columns else {}
    name_map = rank.set_index("_TickerNorm")["Nombre"].to_dict() if "Nombre" in rank.columns else {}

    rows = []
    for ticker_norm, pos in positions.items():
        qty = float(pos.get("quantity", 0.0) or 0.0)
        avg_price = float(pos.get("avg_price", 0.0) or 0.0)
        if qty <= 0:
            continue

        current_price = float(close_map.get(ticker_norm, avg_price))
        capital_actual = qty * current_price

        rows.append({
            "Ticker": optimizer.ticker_para_output(ticker_norm, market),
            "_TickerNorm": ticker_norm,
            "Nombre": name_map.get(ticker_norm, ticker_norm),
            "Quantity": qty,
            "Avg_Price": avg_price,
            "Current_Price": current_price,
            "Capital_Actual": capital_actual,
            "Score_Actual": float(score_map.get(ticker_norm, neutral_score)),
            "Señal_Actual": signal_map.get(ticker_norm, ""),
        })

    return pd.DataFrame(rows)


def _affordable_buy_notional(cash: float, params: Dict) -> float:
    costs = params["costs"]
    pct = (costs["buy_fee_pct"] + costs["slippage_pct"]) / 100.0
    fixed = float(costs["fixed_cost_per_trade"])

    if cash <= fixed:
        return 0.0

    gross = (cash - fixed) / (1.0 + pct)
    return max(0.0, gross)


def _portfolio_value(positions: Dict[str, Dict[str, float]], close_map: Dict[str, float], cash: float) -> float:
    total = float(cash)
    for ticker_norm, pos in positions.items():
        qty = float(pos.get("quantity", 0.0) or 0.0)
        if qty <= 0:
            continue
        px = float(close_map.get(ticker_norm, pos.get("last_price", pos.get("avg_price", 0.0))))
        total += qty * px
    return total


def _simulate_scenario(
    market: str,
    scenario_id: str,
    target_days: int,
    target_min_return: float,
    stop_loss_pct: float,
    cfg: SimulationConfig,
    ranking_all_days: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict]:
    params = optimizer.cargar_parametros_portfolio(market)
    params = dict(params)
    params["capital_inicial"] = float(cfg.initial_capital)
    params["stop_loss_pct"] = float(stop_loss_pct)

    cash = float(cfg.initial_capital)
    positions: Dict[str, Dict[str, float]] = {}

    details = []
    wins = 0
    losses = 0

    trading_days = sorted(ranking_all_days["Date"].dropna().unique())

    for fecha in trading_days:
        ranking_day = ranking_all_days[ranking_all_days["Date"] == fecha].copy()
        ranking_day.sort_values("Score_Final", ascending=False, inplace=True)
        ranking_day["_TickerNorm"] = ranking_day["Ticker"].apply(
            lambda x: optimizer.ticker_para_lookup(x, market)
        )

        close_map = ranking_day.drop_duplicates("_TickerNorm", keep="first").set_index("_TickerNorm")["Close"].to_dict()
        score_map = ranking_day.drop_duplicates("_TickerNorm", keep="first").set_index("_TickerNorm")["Score_Final"].to_dict()
        signal_map = ranking_day.drop_duplicates("_TickerNorm", keep="first").set_index("_TickerNorm")["Señal"].to_dict() if "Señal" in ranking_day.columns else {}

        stopped_tickers = set()
        if stop_loss_pct > 0:
            for ticker_norm, pos in list(positions.items()):
                qty = float(pos.get("quantity", 0.0) or 0.0)
                avg_price = float(pos.get("avg_price", 0.0) or 0.0)
                if qty <= 0 or avg_price <= 0:
                    continue

                price = float(close_map.get(ticker_norm, pos.get("last_price", avg_price) or avg_price))
                stop_price = avg_price * (1.0 - stop_loss_pct)
                pos["last_price"] = price

                if price <= 0 or price > stop_price:
                    continue

                gross = qty * price
                cost, _ = optimizer.calcular_costo_operacion(gross, "SELL", params)
                net_cash = gross - cost
                pnl = net_cash - qty * avg_price

                cash += net_cash
                positions.pop(ticker_norm, None)
                stopped_tickers.add(ticker_norm)

                if pnl > 0:
                    wins += 1
                    wl = "WIN"
                elif pnl < 0:
                    losses += 1
                    wl = "LOSS"
                else:
                    wl = "FLAT"

                running_total = _portfolio_value(positions, close_map, cash)
                details.append({
                    "Row_Type": "TRADE",
                    "Scenario_ID": scenario_id,
                    "Market": market,
                    "Date": pd.Timestamp(fecha).date().isoformat(),
                    "Target_Days": target_days,
                    "Target_Min_Return": target_min_return,
                    "Stop_Loss_Pct": stop_loss_pct,
                    "Action": "STOP_SELL",
                    "Ticker": optimizer.ticker_para_output(ticker_norm, market),
                    "Quantity": round(qty, 10),
                    "Price": round(price, 8),
                    "Gross": round(gross, 2),
                    "Cost": round(cost, 2),
                    "Net_Cash_Flow": round(net_cash, 2),
                    "Transaction_PnL": round(pnl, 2),
                    "Win_Loss": wl,
                    "Running_Cash": round(cash, 2),
                    "Running_Portfolio": round(running_total, 2),
                    "Suggested_Action": "STOP_LIMIT_EXIT",
                    "Signal": signal_map.get(ticker_norm, ""),
                    "Score_Final": float(score_map.get(ticker_norm, 0.0) or 0.0),
                    "Comment": (
                        f"Stop-loss ejecutado: precio {price:.8f} <= límite {stop_price:.8f} "
                        f"({stop_loss_pct * 100:.2f}% debajo del promedio)."
                    ),
                })

        ranking_day_plan = ranking_day[~ranking_day["_TickerNorm"].isin(stopped_tickers)].copy()

        cartera_actual = _build_cartera_actual(
            positions,
            ranking_day_plan,
            market,
            neutral_score=float(params.get("neutral_score", 50.0)),
        )

        decision = {
            "Max_Posiciones": int(cfg.max_positions),
            "Fecha": str(pd.Timestamp(fecha).date()),
        }

        cartera_obj = optimizer.construir_cartera_objetivo(
            decision=decision,
            ranking=ranking_day_plan,
            cartera_actual=cartera_actual,
            params=params,
            nombre_mercado=market,
        )

        plan = optimizer.generar_plan_rebalanceo(
            cartera_objetivo=cartera_obj,
            cartera_actual=cartera_actual,
            ranking=ranking_day_plan,
            params=params,
            capital_total=_portfolio_value(positions, close_map, cash),
            nombre_mercado=market,
            cash_actual=cash,
        )

        if plan.empty:
            running_total = _portfolio_value(positions, close_map, cash)
            details.append({
                "Row_Type": "SNAPSHOT",
                "Scenario_ID": scenario_id,
                "Market": market,
                "Date": pd.Timestamp(fecha).date().isoformat(),
                "Target_Days": target_days,
                "Target_Min_Return": target_min_return,
                "Stop_Loss_Pct": stop_loss_pct,
                "Action": "NO_TRADE",
                "Ticker": "",
                "Quantity": 0.0,
                "Price": 0.0,
                "Gross": 0.0,
                "Cost": 0.0,
                "Net_Cash_Flow": 0.0,
                "Transaction_PnL": 0.0,
                "Win_Loss": "NA",
                "Running_Cash": round(cash, 2),
                "Running_Portfolio": round(running_total, 2),
                "Suggested_Action": "NO_PLAN",
                "Signal": "",
                "Score_Final": 0.0,
                "Comment": "Sin rebalanceo sugerido.",
            })
            continue

        sells = plan[(plan["Ejecutar"] == True) & (plan["Delta_Capital"] < 0)].copy()  # noqa: E712
        buys = plan[(plan["Ejecutar"] == True) & (plan["Delta_Capital"] > 0)].copy()  # noqa: E712

        for _, row in sells.sort_values("Delta_Capital").iterrows():
            ticker_norm = optimizer.ticker_para_lookup(row.get("Ticker", ""), market)
            pos = positions.get(ticker_norm)
            if not pos:
                continue

            price = float(close_map.get(ticker_norm, row.get("Close", 0.0) or 0.0))
            qty_hold = float(pos.get("quantity", 0.0) or 0.0)
            if price <= 0 or qty_hold <= 0:
                continue

            target_notional = min(abs(float(row.get("Delta_Capital", 0.0) or 0.0)), qty_hold * price)
            qty = min(qty_hold, target_notional / price)
            if qty <= 0:
                continue

            gross = qty * price
            cost, _ = optimizer.calcular_costo_operacion(gross, "SELL", params)
            net_cash = gross - cost
            pnl = net_cash - qty * float(pos.get("avg_price", 0.0) or 0.0)

            cash += net_cash
            pos["quantity"] = qty_hold - qty
            pos["last_price"] = price

            if pos["quantity"] <= 1e-10:
                positions.pop(ticker_norm, None)

            if pnl > 0:
                wins += 1
                wl = "WIN"
            elif pnl < 0:
                losses += 1
                wl = "LOSS"
            else:
                wl = "FLAT"

            running_total = _portfolio_value(positions, close_map, cash)
            details.append({
                "Row_Type": "TRADE",
                "Scenario_ID": scenario_id,
                "Market": market,
                "Date": pd.Timestamp(fecha).date().isoformat(),
                "Target_Days": target_days,
                "Target_Min_Return": target_min_return,
                "Stop_Loss_Pct": stop_loss_pct,
                "Action": "SELL",
                "Ticker": optimizer.ticker_para_output(ticker_norm, market),
                "Quantity": round(qty, 10),
                "Price": round(price, 8),
                "Gross": round(gross, 2),
                "Cost": round(cost, 2),
                "Net_Cash_Flow": round(net_cash, 2),
                "Transaction_PnL": round(pnl, 2),
                "Win_Loss": wl,
                "Running_Cash": round(cash, 2),
                "Running_Portfolio": round(running_total, 2),
                "Suggested_Action": row.get("Accion_Sugerida", ""),
                "Signal": row.get("Señal_Actual", ""),
                "Score_Final": float(row.get("Score_Actual", 0.0) or 0.0),
                "Comment": row.get("Motivo", ""),
            })

        for _, row in buys.sort_values("Score_Objetivo", ascending=False).iterrows():
            ticker_norm = optimizer.ticker_para_lookup(row.get("Ticker", ""), market)
            if not ticker_norm:
                continue

            price = float(close_map.get(ticker_norm, row.get("Close", 0.0) or 0.0))
            if price <= 0:
                continue

            desired = float(row.get("Delta_Capital", 0.0) or 0.0)
            affordable = _affordable_buy_notional(cash, params)
            gross = min(desired, affordable)
            if gross <= 0:
                continue

            cost, _ = optimizer.calcular_costo_operacion(gross, "BUY", params)
            total_out = gross + cost
            if total_out > cash:
                continue

            qty = gross / price
            if qty <= 0:
                continue

            pos = positions.get(ticker_norm, {"quantity": 0.0, "avg_price": 0.0, "last_price": price})
            prev_qty = float(pos.get("quantity", 0.0) or 0.0)
            prev_avg = float(pos.get("avg_price", 0.0) or 0.0)
            new_qty = prev_qty + qty

            if new_qty > 0:
                new_avg = ((prev_qty * prev_avg) + (qty * price)) / new_qty
            else:
                new_avg = price

            positions[ticker_norm] = {
                "quantity": new_qty,
                "avg_price": new_avg,
                "last_price": price,
            }

            cash -= total_out
            running_total = _portfolio_value(positions, close_map, cash)
            details.append({
                "Row_Type": "TRADE",
                "Scenario_ID": scenario_id,
                "Market": market,
                "Date": pd.Timestamp(fecha).date().isoformat(),
                "Target_Days": target_days,
                "Target_Min_Return": target_min_return,
                "Stop_Loss_Pct": stop_loss_pct,
                "Action": "BUY",
                "Ticker": optimizer.ticker_para_output(ticker_norm, market),
                "Quantity": round(qty, 10),
                "Price": round(price, 8),
                "Gross": round(gross, 2),
                "Cost": round(cost, 2),
                "Net_Cash_Flow": round(-total_out, 2),
                "Transaction_PnL": 0.0,
                "Win_Loss": "OPEN",
                "Running_Cash": round(cash, 2),
                "Running_Portfolio": round(running_total, 2),
                "Suggested_Action": row.get("Accion_Sugerida", ""),
                "Signal": row.get("Señal_Objetivo", ""),
                "Score_Final": float(row.get("Score_Objetivo", 0.0) or 0.0),
                "Comment": row.get("Motivo", ""),
            })

        running_total = _portfolio_value(positions, close_map, cash)
        details.append({
            "Row_Type": "SNAPSHOT",
            "Scenario_ID": scenario_id,
            "Market": market,
            "Date": pd.Timestamp(fecha).date().isoformat(),
            "Target_Days": target_days,
            "Target_Min_Return": target_min_return,
            "Stop_Loss_Pct": stop_loss_pct,
            "Action": "DAILY_CLOSE",
            "Ticker": "",
            "Quantity": 0.0,
            "Price": 0.0,
            "Gross": 0.0,
            "Cost": 0.0,
            "Net_Cash_Flow": 0.0,
            "Transaction_PnL": 0.0,
            "Win_Loss": "NA",
            "Running_Cash": round(cash, 2),
            "Running_Portfolio": round(running_total, 2),
            "Suggested_Action": "",
            "Signal": "",
            "Score_Final": 0.0,
            "Comment": "Valuacion diaria luego de ejecutar el plan.",
        })

    liquidation_date = pd.Timestamp(cfg.liquidation_date)
    liq_data = ranking_all_days[ranking_all_days["Date"] <= liquidation_date].copy()
    liq_data.sort_values(["Ticker", "Date"], inplace=True)
    liq_last = liq_data.groupby("Ticker", as_index=False).tail(1)
    liq_last["_TickerNorm"] = liq_last["Ticker"].apply(lambda x: optimizer.ticker_para_lookup(x, market))
    liq_close_map = liq_last.set_index("_TickerNorm")["Close"].to_dict() if not liq_last.empty else {}

    for ticker_norm, pos in list(positions.items()):
        qty = float(pos.get("quantity", 0.0) or 0.0)
        if qty <= 0:
            continue

        price = float(liq_close_map.get(ticker_norm, pos.get("last_price", pos.get("avg_price", 0.0))))
        gross = qty * price
        cost, _ = optimizer.calcular_costo_operacion(gross, "SELL", params)
        net_cash = gross - cost
        pnl = net_cash - qty * float(pos.get("avg_price", 0.0) or 0.0)

        cash += net_cash
        positions.pop(ticker_norm, None)

        if pnl > 0:
            wins += 1
            wl = "WIN"
        elif pnl < 0:
            losses += 1
            wl = "LOSS"
        else:
            wl = "FLAT"

        details.append({
            "Row_Type": "TRADE",
            "Scenario_ID": scenario_id,
            "Market": market,
            "Date": liquidation_date.date().isoformat(),
            "Target_Days": target_days,
            "Target_Min_Return": target_min_return,
            "Stop_Loss_Pct": stop_loss_pct,
            "Action": "FORCED_SELL",
            "Ticker": optimizer.ticker_para_output(ticker_norm, market),
            "Quantity": round(qty, 10),
            "Price": round(price, 8),
            "Gross": round(gross, 2),
            "Cost": round(cost, 2),
            "Net_Cash_Flow": round(net_cash, 2),
            "Transaction_PnL": round(pnl, 2),
            "Win_Loss": wl,
            "Running_Cash": round(cash, 2),
            "Running_Portfolio": round(cash, 2),
            "Suggested_Action": "FORCED_EXIT",
            "Signal": "",
            "Score_Final": 0.0,
            "Comment": "Liquidacion forzada al cierre de la fecha final.",
        })

    total_trades = sum(1 for r in details if r["Row_Type"] == "TRADE")
    final_value = float(cash)
    margin_pct = ((final_value / cfg.initial_capital) - 1.0) * 100.0
    win_rate = (wins / (wins + losses) * 100.0) if (wins + losses) > 0 else 0.0

    summary = {
        "Scenario_ID": scenario_id,
        "Market": market,
        "Target_Days": target_days,
        "Target_Min_Return": target_min_return,
        "Stop_Loss_Pct": stop_loss_pct,
        "Start_Date": cfg.start_date,
        "End_Date": cfg.end_date,
        "Liquidation_Date": cfg.liquidation_date,
        "Initial_Capital": round(cfg.initial_capital, 2),
        "Final_Portfolio": round(final_value, 2),
        "Net_Result": round(final_value - cfg.initial_capital, 2),
        "Margin_Pct": round(margin_pct, 4),
        "Trades": int(total_trades),
        "Wins": int(wins),
        "Losses": int(losses),
        "Win_Rate_Pct": round(win_rate, 2),
    }

    return pd.DataFrame(details), summary


def run_optimizer_simulations(cfg: SimulationConfig | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg = cfg or SimulationConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    start_ts = pd.Timestamp(cfg.start_date)
    end_ts = pd.Timestamp(cfg.end_date)

    scenario_summaries = []
    detail_frames = []

    for market, target_days, min_return, stop_loss_pct in product(
        cfg.markets,
        cfg.target_days_list,
        cfg.target_min_returns,
        cfg.stop_loss_pct_list,
    ):
        sid = _scenario_id(market, target_days, min_return, stop_loss_pct)
        print("=" * 90)
        print(f"SIMULACION: {sid}")
        print("=" * 90)

        try:
            if cfg.run_pipeline_per_scenario:
                _prepare_scenario_artifacts(market, target_days, min_return)

            ranking_all_days = _build_daily_ranking_frame(
                market=market,
                start_date=start_ts,
                end_date=end_ts,
                top_models=cfg.top_models_for_scoring,
            )

            detail_df, summary = _simulate_scenario(
                market=market,
                scenario_id=sid,
                target_days=target_days,
                target_min_return=min_return,
                stop_loss_pct=stop_loss_pct,
                cfg=cfg,
                ranking_all_days=ranking_all_days,
            )

            market_out_dir = cfg.output_dir / market
            market_out_dir.mkdir(parents=True, exist_ok=True)

            detail_name = (
                f"sim_{market}_t{target_days}d_r{_ret_label(min_return)}_sl{_stop_label(stop_loss_pct)}_"
                f"{pd.Timestamp(cfg.start_date).date()}_{pd.Timestamp(cfg.liquidation_date).date()}.csv"
            )
            detail_path = market_out_dir / detail_name
            detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

            summary["Detail_File"] = str(detail_path)
            scenario_summaries.append(summary)
            detail_frames.append(detail_df)

            print(f"Detalle escenario: {detail_path}")
            print(f"Capital final: ${summary['Final_Portfolio']:,.2f} | Margen: {summary['Margin_Pct']:.2f}%")

        except Exception as e:
            scenario_summaries.append({
                "Scenario_ID": sid,
                "Market": market,
                "Target_Days": target_days,
                "Target_Min_Return": min_return,
                "Stop_Loss_Pct": stop_loss_pct,
                "Start_Date": cfg.start_date,
                "End_Date": cfg.end_date,
                "Liquidation_Date": cfg.liquidation_date,
                "Initial_Capital": cfg.initial_capital,
                "Final_Portfolio": np.nan,
                "Net_Result": np.nan,
                "Margin_Pct": np.nan,
                "Trades": 0,
                "Wins": 0,
                "Losses": 0,
                "Win_Rate_Pct": 0.0,
                "Detail_File": "",
                "Error": str(e),
            })
            print(f"ERROR en {sid}: {e}")

    summary_df = pd.DataFrame(scenario_summaries)
    summary_path = cfg.output_dir / "simulacion_resumen_global.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print("=" * 90)
    print(f"Resumen global: {summary_path}")
    print("=" * 90)

    if detail_frames:
        details_all_df = pd.concat(detail_frames, ignore_index=True)
    else:
        details_all_df = pd.DataFrame()

    return summary_df, details_all_df


if __name__ == "__main__":
    run_optimizer_simulations()
