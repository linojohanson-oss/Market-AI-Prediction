from __future__ import annotations

import argparse
from typing import Callable, List, Optional

from source.optimizer_simulator import SimulationConfig, run_optimizer_simulations


def _parse_csv_list(raw: Optional[str], caster: Callable[[str], object]) -> Optional[List]:
    if raw is None:
        return None

    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    if not parts:
        return None

    values = []
    for part in parts:
        values.append(caster(part))
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run optimizer simulations without inline Python one-liners.",
    )

    parser.add_argument("--markets", help="Comma-separated markets. Example: crypto,nasdaq")
    parser.add_argument("--target-days", help="Comma-separated target days. Example: 5,10,15")
    parser.add_argument(
        "--target-min-returns",
        help="Comma-separated target min returns. Example: 0.015,0.03,0.05",
    )
    parser.add_argument(
        "--stop-loss-pcts",
        help="Comma-separated stop-loss percentages. Example: 0.015,0.02,0.03",
    )
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--liquidation-date", help="Liquidation date (YYYY-MM-DD)")
    parser.add_argument("--initial-capital", type=float, help="Initial capital")
    parser.add_argument("--max-positions", type=int, help="Max positions")
    parser.add_argument("--top-models", type=int, help="Top models used for scoring")
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Skip data/feature/training pipeline for each scenario and reuse existing artifacts.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg_kwargs = {}

    markets = _parse_csv_list(args.markets, str)
    if markets is not None:
        cfg_kwargs["markets"] = markets

    target_days = _parse_csv_list(args.target_days, int)
    if target_days is not None:
        cfg_kwargs["target_days_list"] = target_days

    min_returns = _parse_csv_list(args.target_min_returns, float)
    if min_returns is not None:
        cfg_kwargs["target_min_returns"] = min_returns

    stop_loss_pcts = _parse_csv_list(args.stop_loss_pcts, float)
    if stop_loss_pcts is not None:
        cfg_kwargs["stop_loss_pct_list"] = stop_loss_pcts

    if args.start_date:
        cfg_kwargs["start_date"] = args.start_date
    if args.end_date:
        cfg_kwargs["end_date"] = args.end_date
    if args.liquidation_date:
        cfg_kwargs["liquidation_date"] = args.liquidation_date
    if args.initial_capital is not None:
        cfg_kwargs["initial_capital"] = args.initial_capital
    if args.max_positions is not None:
        cfg_kwargs["max_positions"] = args.max_positions
    if args.top_models is not None:
        cfg_kwargs["top_models_for_scoring"] = args.top_models
    if args.skip_pipeline:
        cfg_kwargs["run_pipeline_per_scenario"] = False

    cfg = SimulationConfig(**cfg_kwargs)
    run_optimizer_simulations(cfg)


if __name__ == "__main__":
    main()
