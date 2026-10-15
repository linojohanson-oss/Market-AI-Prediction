import os

import pandas as pd

from source.portfolio_optimizer import generar_plan_rebalanceo


def _base_params(stop_loss_pct=0.015):
    return {
        "neutral_score": 50.0,
        "alpha_per_score_point_pct": 0.03,
        "min_edge_buffer_pct": 0.2,
        "min_trade_value": 150.0,
        "stop_loss_pct": stop_loss_pct,
        "costs": {
            "buy_fee_pct": 0.1,
            "sell_fee_pct": 0.1,
            "slippage_pct": 0.05,
            "tax_sell_pct": 0.0,
            "fixed_cost_per_trade": 0.0,
        },
    }


def test_stop_loss_forces_exit_even_without_buys():
    ranking = pd.DataFrame(
        {
            "Ticker": ["BTC-USDT"],
            "Nombre": ["Bitcoin"],
            "Close": [95.0],
            "Score_Final": [48.0],
            "Señal": ["NEUTRAL"],
        }
    )
    cartera_actual = pd.DataFrame(
        {
            "Ticker": ["BTC-USDT"],
            "_TickerNorm": ["BTC-USD"],
            "Nombre": ["Bitcoin"],
            "Quantity": [1.0],
            "Avg_Price": [100.0],
            "Current_Price": [95.0],
            "Capital_Actual": [95.0],
            "Score_Actual": [48.0],
            "Señal_Actual": ["NEUTRAL"],
        }
    )

    plan = generar_plan_rebalanceo(
        cartera_objetivo=pd.DataFrame(),
        cartera_actual=cartera_actual,
        ranking=ranking,
        params=_base_params(stop_loss_pct=0.015),
        capital_total=95.0,
        nombre_mercado="crypto",
        cash_actual=0.0,
    )

    assert len(plan) == 1
    assert plan.iloc[0]["Accion_Deseada"] == "SELL"
    assert plan.iloc[0]["Accion_Sugerida"] == "STOP_LOSS_EXIT"
    assert bool(plan.iloc[0]["Ejecutar"]) is True
    assert bool(plan.iloc[0]["Stop_Loss_Triggered"]) is True


def test_stop_loss_does_not_trigger_above_threshold():
    ranking = pd.DataFrame(
        {
            "Ticker": ["BTC-USDT"],
            "Nombre": ["Bitcoin"],
            "Close": [99.0],
            "Score_Final": [48.0],
            "Señal": ["NEUTRAL"],
        }
    )
    cartera_actual = pd.DataFrame(
        {
            "Ticker": ["BTC-USDT"],
            "_TickerNorm": ["BTC-USD"],
            "Nombre": ["Bitcoin"],
            "Quantity": [1.0],
            "Avg_Price": [100.0],
            "Current_Price": [99.0],
            "Capital_Actual": [99.0],
            "Score_Actual": [48.0],
            "Señal_Actual": ["NEUTRAL"],
        }
    )

    plan = generar_plan_rebalanceo(
        cartera_objetivo=pd.DataFrame(),
        cartera_actual=cartera_actual,
        ranking=ranking,
        params=_base_params(stop_loss_pct=0.015),
        capital_total=99.0,
        nombre_mercado="crypto",
        cash_actual=0.0,
    )

    assert len(plan) == 1
    assert plan.iloc[0]["Accion_Sugerida"] == "HOLD"
    assert bool(plan.iloc[0]["Ejecutar"]) is False
    assert bool(plan.iloc[0]["Stop_Loss_Triggered"]) is False


def test_plan_includes_live_price_next_to_close():
    ranking = pd.DataFrame(
        {
            "Ticker": ["BTC-USDT"],
            "Nombre": ["Bitcoin"],
            "Close": [95.0],
            "Score_Final": [55.0],
            "Señal": ["BUY"],
        }
    )
    cartera_actual = pd.DataFrame(
        {
            "Ticker": ["BTC-USDT"],
            "_TickerNorm": ["BTC-USD"],
            "Nombre": ["Bitcoin"],
            "Quantity": [1.0],
            "Avg_Price": [100.0],
            "Current_Price": [98.5],
            "Capital_Actual": [98.5],
            "Score_Actual": [50.0],
            "Señal_Actual": ["BUY"],
        }
    )

    plan = generar_plan_rebalanceo(
        cartera_objetivo=pd.DataFrame(),
        cartera_actual=cartera_actual,
        ranking=ranking,
        params=_base_params(stop_loss_pct=0.015),
        capital_total=98.5,
        nombre_mercado="crypto",
        cash_actual=0.0,
    )

    assert "Current_Price" in plan.columns
    assert list(plan.columns).index("Current_Price") == list(plan.columns).index("Close") + 1
    assert plan.iloc[0]["Current_Price"] == 98.5


def test_live_price_reduces_buy_edge_when_price_already_above_close():
    os.environ["MARKET_LIVE_PRICES"] = "1"
    try:
        ranking = pd.DataFrame(
            {
                "Ticker": ["BTC-USDT"],
                "Nombre": ["Bitcoin"],
                "Close": [100.0],
                "Score_Final": [75.0],
                "Señal": ["BUY"],
            }
        )
        cartera_actual = pd.DataFrame(
            {
                "Ticker": ["BTC-USDT"],
                "_TickerNorm": ["BTC-USD"],
                "Nombre": ["Bitcoin"],
                "Quantity": [1.0],
                "Avg_Price": [100.0],
                "Current_Price": [100.05],
                "Capital_Actual": [100.05],
                "Score_Actual": [50.0],
                "Señal_Actual": ["NEUTRAL"],
            }
        )
        cartera_objetivo = pd.DataFrame(
            [
                {
                    "Ticker": "BTC-USDT",
                    "Nombre": "Bitcoin",
                    "Close": 100.0,
                    "Capital": 105.0,
                    "Score_Final": 75.0,
                    "Señal": "BUY",
                    "_TickerNorm": "BTC-USD",
                    "es_actual": False,
                }
            ]
        )

        plan = generar_plan_rebalanceo(
            cartera_objetivo=cartera_objetivo,
            cartera_actual=cartera_actual,
            ranking=ranking,
            params=_base_params(stop_loss_pct=0.015),
            capital_total=120.05,
            nombre_mercado="crypto",
            cash_actual=20.0,
        )

        assert plan.iloc[0]["Accion_Deseada"] == "BUY"
        assert bool(plan.iloc[0]["Ejecutar"]) is False
        assert plan.iloc[0]["Ventaja_Neta_pct"] <= 0
    finally:
        os.environ.pop("MARKET_LIVE_PRICES", None)


def test_non_live_mode_keeps_historical_close_baseline():
    os.environ.pop("MARKET_LIVE_PRICES", None)
    ranking = pd.DataFrame(
        {
            "Ticker": ["BTC-USDT"],
            "Nombre": ["Bitcoin"],
            "Close": [100.0],
            "Score_Final": [75.0],
            "Señal": ["BUY"],
        }
    )
    cartera_actual = pd.DataFrame(
        {
            "Ticker": ["BTC-USDT"],
            "_TickerNorm": ["BTC-USD"],
            "Nombre": ["Bitcoin"],
            "Quantity": [1.0],
            "Avg_Price": [100.0],
            "Current_Price": [100.05],
            "Capital_Actual": [100.05],
            "Score_Actual": [50.0],
            "Señal_Actual": ["NEUTRAL"],
        }
    )
    cartera_objetivo = pd.DataFrame(
        [
            {
                "Ticker": "BTC-USDT",
                "Nombre": "Bitcoin",
                "Close": 100.0,
                "Capital": 105.0,
                "Score_Final": 75.0,
                "Señal": "BUY",
                "_TickerNorm": "BTC-USD",
                "es_actual": False,
            }
        ]
    )

    plan = generar_plan_rebalanceo(
        cartera_objetivo=cartera_objetivo,
        cartera_actual=cartera_actual,
        ranking=ranking,
        params=_base_params(stop_loss_pct=0.015),
        capital_total=120.05,
        nombre_mercado="crypto",
        cash_actual=20.0,
    )

    assert plan.iloc[0]["Accion_Deseada"] == "BUY"
    assert bool(plan.iloc[0]["Ejecutar"]) is True
    assert plan.iloc[0]["Ventaja_Neta_pct"] > 0