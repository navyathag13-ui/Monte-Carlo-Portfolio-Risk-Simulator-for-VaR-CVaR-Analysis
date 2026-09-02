"""
main.py
=======
Orchestrator for the Monte Carlo Portfolio Risk Simulator.

Run with:
    python main.py

Optional CLI overrides:
    python main.py --tickers SPY QQQ TLT --weights 0.5 0.3 0.2 --value 500000
    python main.py --sims 20000 --seed 7 --no-plots
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

import config
from data_loader import load_price_data
from portfolio import build_portfolio
from simulation import run_all_horizons
from risk_metrics import (
    build_risk_table,
    distribution_stats,
    risk_at_confidence,
)
from stress_testing import run_all_stress_scenarios, worst_scenario
from visualization import (
    plot_return_distribution,
    plot_pnl_distribution,
    plot_correlation_heatmap,
    plot_cumulative_returns,
    plot_simulation_paths,
    plot_stress_comparison,
    plot_multi_horizon_var,
    plot_component_var,
)
from utils import (
    get_logger,
    ensure_dirs,
    save_dataframe,
    build_percentile_table,
)

logger = get_logger("main")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Monte Carlo Portfolio Risk Simulator — VaR / CVaR / Stress Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples
            --------
            # Default portfolio (SPY QQQ TLT GLD AAPL)
            python main.py

            # Custom portfolio
            python main.py --tickers SPY QQQ GLD --weights 0.50 0.30 0.20

            # 500 k book, 20 k simulations, no charts
            python main.py --value 500000 --sims 20000 --no-plots
        """),
    )
    p.add_argument("--tickers",  nargs="+", default=None,
                   metavar="TICKER",
                   help="Space-separated ticker symbols (e.g. SPY QQQ TLT).")
    p.add_argument("--weights",  nargs="+", type=float, default=None,
                   metavar="W",
                   help="Portfolio weights — must match ticker count and sum to 1.")
    p.add_argument("--value",    type=float, default=None, metavar="USD",
                   help="Portfolio notional value in USD (default: 1,000,000).")
    p.add_argument("--sims",     type=int,   default=None, metavar="N",
                   help="Number of Monte Carlo paths (default: 10,000).")
    p.add_argument("--seed",     type=int,   default=None, metavar="INT",
                   help="Random seed for full reproducibility.")
    p.add_argument("--start",    type=str,   default=None, metavar="YYYY-MM-DD",
                   help="Historical data start date.")
    p.add_argument("--end",      type=str,   default=None, metavar="YYYY-MM-DD",
                   help="Historical data end date.")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip chart generation (useful for CI / fast runs).")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # ------------------------------------------------------------------
    # 0. Resolve configuration (CLI overrides config.py defaults)
    # ------------------------------------------------------------------
    tickers = args.tickers or config.TICKERS
    weights = args.weights or config.WEIGHTS
    value   = args.value   or config.PORTFOLIO_VALUE
    n_sims  = args.sims    or config.N_SIMULATIONS
    seed    = args.seed    or config.RANDOM_SEED
    start   = args.start   or config.DATA_START
    end     = args.end     or config.DATA_END
    generate_plots = not args.no_plots

    ensure_dirs(config.CHARTS_DIR, config.TABLES_DIR, config.SUMMARIES_DIR)

    logger.info("=" * 60)
    logger.info("  Monte Carlo Portfolio Risk Simulator")
    logger.info("=" * 60)
    logger.info("Tickers  : %s", tickers)
    logger.info("Weights  : %s", weights)
    logger.info("Value    : $%s", f"{value:,.0f}")
    logger.info("Sims     : %d", n_sims)
    logger.info("Seed     : %d", seed)

    # ------------------------------------------------------------------
    # 1. Load price data
    # ------------------------------------------------------------------
    prices = load_price_data(tickers, start, end, random_seed=seed)

    # ------------------------------------------------------------------
    # 2. Build portfolio
    # ------------------------------------------------------------------
    portfolio = build_portfolio(tickers, weights, prices, portfolio_value=value)
    portfolio.print_summary()

    save_dataframe(portfolio.returns, f"{config.TABLES_DIR}/daily_returns.csv")

    # ------------------------------------------------------------------
    # 3. Monte Carlo simulations — all configured horizons
    # ------------------------------------------------------------------
    logger.info("Running Monte Carlo simulations …")
    horizon_results = run_all_horizons(
        portfolio=portfolio,
        horizons=config.HORIZONS,
        n_simulations=n_sims,
        random_seed=seed,
        full_paths_horizon="252-day",
    )

    # ------------------------------------------------------------------
    # 4. Risk metrics
    # ------------------------------------------------------------------
    hist_returns  = portfolio.historical_portfolio_returns().values
    mc_1d_returns = horizon_results["1-day"].portfolio_returns

    risk_df = build_risk_table(
        mc_returns=mc_1d_returns,
        hist_returns=hist_returns,
        confidence_levels=config.CONFIDENCE_LEVELS,
        portfolio_value=value,
    )
    save_dataframe(risk_df, f"{config.TABLES_DIR}/risk_metrics.csv", index=False)

    r95 = risk_at_confidence(mc_1d_returns, 0.95, value)
    r99 = risk_at_confidence(mc_1d_returns, 0.99, value)

    perc_df = build_percentile_table(mc_1d_returns)
    save_dataframe(perc_df, f"{config.TABLES_DIR}/return_percentiles.csv", index=False)

    dist_stats = distribution_stats(mc_1d_returns)
    dist_df = pd.DataFrame([dist_stats]).T.rename(columns={0: "Value"})
    save_dataframe(dist_df, f"{config.TABLES_DIR}/distribution_stats.csv")

    # Component VaR attribution
    comp_var = portfolio.component_var_contributions(r95["var_pct"])
    comp_df  = comp_var.to_frame("VaR Contribution ($)")
    comp_df["VaR Contribution (%)"] = comp_df["VaR Contribution ($)"] / r95["var_usd"] * 100
    comp_df["VaR Contribution (%)"] = comp_df["VaR Contribution (%)"].map("{:.1f}%".format)
    comp_df["VaR Contribution ($)"] = comp_df["VaR Contribution ($)"].map("${:,.0f}".format)
    save_dataframe(comp_df, f"{config.TABLES_DIR}/component_var.csv")

    # ------------------------------------------------------------------
    # 5. Stress testing
    # ------------------------------------------------------------------
    logger.info("Running stress scenarios …")
    scenario_df = run_all_stress_scenarios(
        portfolio=portfolio,
        scenarios=config.STRESS_SCENARIOS,
        confidence_levels=config.CONFIDENCE_LEVELS,
        horizon=1,
        n_simulations=n_sims,
        random_seed=seed,
        portfolio_value=value,
    )
    save_dataframe(scenario_df, f"{config.TABLES_DIR}/stress_scenarios.csv", index=False)

    worst = worst_scenario(scenario_df, confidence="95%", metric="CVaR (%)")

    # ------------------------------------------------------------------
    # 6. Visualisations
    # ------------------------------------------------------------------
    if generate_plots:
        logger.info("Generating charts …")

        plot_return_distribution(
            returns=mc_1d_returns,
            var_95=r95["var_pct"],  var_99=r99["var_pct"],
            cvar_95=r95["cvar_pct"], cvar_99=r99["cvar_pct"],
            horizon_label="1-Day",
            out_path=f"{config.CHARTS_DIR}/01_return_distribution_1d.png",
        )

        plot_pnl_distribution(
            returns=mc_1d_returns,
            portfolio_value=value,
            var_95=r95["var_pct"],
            var_99=r99["var_pct"],
            out_path=f"{config.CHARTS_DIR}/02_pnl_distribution_1d.png",
        )

        plot_correlation_heatmap(
            corr_matrix=portfolio.corr_matrix,
            out_path=f"{config.CHARTS_DIR}/03_correlation_heatmap.png",
        )

        plot_cumulative_returns(
            prices=portfolio.prices,
            weights=portfolio.weights,
            out_path=f"{config.CHARTS_DIR}/04_cumulative_returns.png",
        )

        plot_simulation_paths(
            cumulative_paths=horizon_results["252-day"].cumulative_paths,
            horizon=252,
            n_plot=config.N_PATHS_PLOT,
            portfolio_value=value,
            var_95=r95["var_pct"],
            out_path=f"{config.CHARTS_DIR}/05_simulation_paths_252d.png",
        )

        plot_stress_comparison(
            scenario_df=scenario_df,
            confidence="95%",
            out_path=f"{config.CHARTS_DIR}/06_stress_comparison_95.png",
        )

        plot_multi_horizon_var(
            horizon_results=horizon_results,
            confidence_levels=config.CONFIDENCE_LEVELS,
            portfolio_value=value,
            out_path=f"{config.CHARTS_DIR}/07_multi_horizon_var.png",
        )

        # 10-day return distribution (Basel II capital horizon reference)
        mc_10d  = horizon_results["10-day"]
        r95_10d = risk_at_confidence(mc_10d.portfolio_returns, 0.95, value)
        r99_10d = risk_at_confidence(mc_10d.portfolio_returns, 0.99, value)
        plot_return_distribution(
            returns=mc_10d.portfolio_returns,
            var_95=r95_10d["var_pct"],  var_99=r99_10d["var_pct"],
            cvar_95=r95_10d["cvar_pct"], cvar_99=r99_10d["cvar_pct"],
            horizon_label="10-Day  (Basel II Capital Horizon)",
            out_path=f"{config.CHARTS_DIR}/08_return_distribution_10d.png",
        )

        # Component VaR attribution (new chart)
        plot_component_var(
            component_var_series=portfolio.component_var_contributions(r95["var_pct"]),
            total_var_usd=r95["var_usd"],
            confidence=0.95,
            out_path=f"{config.CHARTS_DIR}/09_component_var_attribution.png",
        )

    # ------------------------------------------------------------------
    # 7. Console interpretation + markdown summary
    # ------------------------------------------------------------------
    _print_interpretation(
        portfolio=portfolio,
        r95=r95, r99=r99,
        scenario_df=scenario_df,
        worst=worst,
        value=value,
        n_sims=n_sims,
        comp_var=portfolio.component_var_contributions(r95["var_pct"]),
    )

    _save_text_summary(
        portfolio=portfolio,
        r95=r95, r99=r99,
        risk_df=risk_df,
        comp_df=comp_df,
        scenario_df=scenario_df,
        worst=worst,
        value=value,
        n_sims=n_sims,
        out_path=f"{config.SUMMARIES_DIR}/risk_summary.md",
    )

    logger.info("All outputs saved to %s/", config.OUTPUT_DIR)
    logger.info("Simulation complete.")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def _print_interpretation(
    portfolio, r95, r99, scenario_df, worst, value, n_sims, comp_var
) -> None:
    sep = "=" * 68
    print(f"\n{sep}")
    print("  RISK ANALYSIS SUMMARY  ·  Monte Carlo Portfolio Risk Simulator")
    print(sep)
    print(f"  Notional Value           : ${value:>15,.0f}")
    print(f"  Simulations              : {n_sims:>15,}")
    print(f"  Historical Observations  : {len(portfolio.returns):>15,}  trading days")
    print(f"  Annualised Return        : {portfolio.annualised_return:>15.2%}")
    print(f"  Annualised Volatility    : {portfolio.annualised_volatility:>15.2%}")
    print(f"  Sharpe Ratio             : {portfolio.sharpe_ratio:>15.3f}  (r_f = {getattr(config, 'RISK_FREE_RATE', 0.04):.0%})")
    print(f"  Max Historical Drawdown  : {portfolio.max_drawdown:>15.2%}")
    print()
    print("  ONE-DAY VALUE AT RISK (VaR)")
    print(f"    95% VaR  : {r95['var_pct']:>9.4%}   │   ${r95['var_usd']:>12,.0f}")
    print(f"    99% VaR  : {r99['var_pct']:>9.4%}   │   ${r99['var_usd']:>12,.0f}")
    print()
    print("  ONE-DAY CONDITIONAL VaR (CVaR  /  Expected Shortfall)")
    print(f"    95% CVaR : {r95['cvar_pct']:>9.4%}   │   ${r95['cvar_usd']:>12,.0f}")
    print(f"    99% CVaR : {r99['cvar_pct']:>9.4%}   │   ${r99['cvar_usd']:>12,.0f}")
    print()
    print("  COMPONENT VaR ATTRIBUTION  (95%  ·  marginal contributions)")
    total_var = r95["var_usd"]
    for ticker, contrib in comp_var.items():
        bar_len = max(1, int(round(contrib / total_var * 30)))
        bar = "█" * bar_len
        print(f"    {ticker:<6} : ${contrib:>8,.0f}  {contrib/total_var:>5.1%}  {bar}")
    print()
    print(f"  WORST STRESS SCENARIO (95 % CVaR basis):  {worst}")
    print()
    print("  PLAIN-ENGLISH INTERPRETATION")
    print(textwrap.fill(
        f"This {len(portfolio.tickers)}-asset portfolio targets "
        f"{portfolio.annualised_return:.1%} annual return at "
        f"{portfolio.annualised_volatility:.1%} volatility "
        f"(Sharpe {portfolio.sharpe_ratio:.2f}).  "
        f"On any single trading day, there is a 5 % chance of losing more than "
        f"{r95['var_pct']:.2%} (${r95['var_usd']:,.0f} on a ${value:,.0f} book) — "
        f"the 95 % VaR figure.  When losses do breach that threshold, the average "
        f"loss is {r95['cvar_pct']:.2%} (${r95['cvar_usd']:,.0f}), the CVaR or "
        f"Expected Shortfall.  "
        f"The Basel II market-risk capital charge uses a 10-day 99 % VaR horizon; "
        f"the FRTB (Basel IV) replaces that with a 97.5 % ES, making CVaR the "
        f"regulatory standard going forward.  "
        f"Under the '{worst}' stress scenario, risk escalates far above the baseline, "
        f"demonstrating the portfolio's tail exposure to correlated market shocks.",
        width=66, initial_indent="  ", subsequent_indent="  ",
    ))
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def _save_text_summary(
    portfolio, r95, r99, risk_df, comp_df, scenario_df,
    worst, value, n_sims, out_path
) -> None:
    ensure_dirs(str(Path(out_path).parent))

    tickers_str = ", ".join(portfolio.tickers)
    weights_str = ", ".join(f"{w:.0%}" for w in portfolio.weights)
    rf = getattr(config, "RISK_FREE_RATE", 0.04)

    lines = [
        "# Monte Carlo Portfolio Risk Simulator — Results Summary",
        "",
        "> Auto-generated report.  All figures use 1-day log-returns unless stated.",
        "",
        "## Portfolio Configuration",
        "| Parameter | Value |",
        "|---|---|",
        f"| **Tickers** | {tickers_str} |",
        f"| **Weights** | {weights_str} |",
        f"| **Notional Value** | ${value:,.0f} |",
        f"| **Simulations** | {n_sims:,} |",
        f"| **Historical Window** | {portfolio.prices.index[0].date()} → {portfolio.prices.index[-1].date()} |",
        f"| **Observations** | {len(portfolio.returns):,} trading days |",
        "",
        "## Portfolio Statistics",
        "| Metric | Value |",
        "|---|---|",
        f"| Annualised Return | **{portfolio.annualised_return:.2%}** |",
        f"| Annualised Volatility | **{portfolio.annualised_volatility:.2%}** |",
        f"| Sharpe Ratio (r_f = {rf:.0%}) | **{portfolio.sharpe_ratio:.3f}** |",
        f"| Max Historical Drawdown | **{portfolio.max_drawdown:.2%}** |",
        f"| Daily Volatility | {portfolio.daily_volatility:.4%} |",
        "",
        "## One-Day Risk Metrics",
        "| Metric | % of Portfolio | USD Amount |",
        "|---|---|---|",
        f"| **95% VaR**  | {r95['var_pct']:.4%}  | **${r95['var_usd']:,.0f}**  |",
        f"| **99% VaR**  | {r99['var_pct']:.4%}  | **${r99['var_usd']:,.0f}**  |",
        f"| **95% CVaR** | {r95['cvar_pct']:.4%} | **${r95['cvar_usd']:,.0f}** |",
        f"| **99% CVaR** | {r99['cvar_pct']:.4%} | **${r99['cvar_usd']:,.0f}** |",
        "",
        "## Method Comparison (1-Day VaR / CVaR)",
        "",
        risk_df.to_markdown(index=False),
        "",
        "## Component VaR Attribution  (95%  ·  marginal contributions)",
        "",
        comp_df.to_markdown(),
        "",
        "> Component VaRs sum to the total portfolio VaR.  "
        "The largest contributor is the primary candidate for a hedge overlay.",
        "",
        "## Stress Scenario Comparison",
        "",
        scenario_df.to_markdown(index=False),
        "",
        f"**Worst stress scenario (95% CVaR):** {worst}",
        "",
        "## Risk Interpretation",
        "",
        f"This portfolio targets **{portfolio.annualised_return:.1%} annual return** "
        f"at **{portfolio.annualised_volatility:.1%} annualised volatility** "
        f"(Sharpe: **{portfolio.sharpe_ratio:.2f}**).",
        "",
        f"- **Daily VaR (95%):** There is a 1-in-20 chance of losing more than "
        f"**{r95['var_pct']:.2%}** (${r95['var_usd']:,.0f}) on a single trading day.",
        f"- **CVaR / Expected Shortfall (95%):** When the loss threshold is breached, "
        f"the expected loss is **{r95['cvar_pct']:.2%}** (${r95['cvar_usd']:,.0f}).  "
        f"This is the metric FRTB (Basel IV) mandates for regulatory capital.",
        f"- **Stress:** The '{worst}' scenario is the most severe, illustrating the "
        f"portfolio's sensitivity to correlated market shocks and volatility regime changes.",
        "",
        "---",
        "_Generated by Monte Carlo Portfolio Risk Simulator_  "
        "·  _Multivariate Normal / Cholesky simulation_",
    ]

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    logger.info("Saved markdown summary → %s", out_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
