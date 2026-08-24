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